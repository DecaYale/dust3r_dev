# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import math
import os
import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl
from plyfile import PlyData, PlyElement
from evo.core.trajectory import PosePath3D, PoseTrajectory3D



pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                         "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                         "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, transform_to_opengl=True,
                                 colmap_c2w_poses=None, draw_cameras=True, export_name='scene.obj'):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    
    if colmap_c2w_poses is not None:
        # import pdb; pdb.set_trace()
        # traj_colmap = PosePath3D(poses_se3=np.linalg.inv(colmap_poses ))#[:,:3,3]) # world2cam->cam2world
        # r_a, t_a, s = traj_colmap.align(traj_ref=traj_dust3r, correct_scale=True)
        try:
            traj_colmap = PosePath3D(poses_se3=colmap_c2w_poses)#[:,:3,3]) # cam2world
            traj_dust3r = PosePath3D(poses_se3=cams2world)#[:,:3,3])
            r_a, t_a, s = traj_dust3r.align(traj_ref=traj_colmap, correct_scale=True)
        except:
            import pdb; pdb.set_trace()
            cams2world[:,:3,3] = cams2world[:,:3,3] + np.random.randn(cams2world.shape[0], 3)*1e-6
            traj_colmap = PosePath3D(poses_se3=colmap_c2w_poses)#[:,:3,3]) # cam2world
            traj_dust3r = PosePath3D(poses_se3=cams2world)#[:,:3,3])
            r_a, t_a, s = traj_dust3r.align(traj_ref=traj_colmap, correct_scale=True)


        align_transform = np.eye(4)
        # align_transform[:3, :3] *= s
        align_transform[:3, :3] = r_a * s #@ align_transform[:3, :3]
        # align_transform[:3, :3] = r_a @ align_transform[:3, :3]
        align_transform[:3, 3] = t_a
        # aligned_colmap_c2w_list= [] 
        # for i in range(len(colmap_c2w_poses)):
        #     # colmap_cam2world = np.linalg.inv(colmap_c2w_poses[i])
        #     colmap_cam2world = colmap_c2w_poses[i]
        #     # colmap_cam2world = colmap_poses[i]
        #     align_transform = np.eye(4)
        #     align_transform[:3, :3] = r_a
        #     align_transform[:3, 3] = t_a

        #     colmap_cam2world[:3,3] = colmap_cam2world[:3,3] * s
        #     aligned_colmap_c2w = align_transform@colmap_cam2world
        #     # aligned_colmap_c2w[:3,3] = aligned_colmap_c2w[:3,3] * s
        #     aligned_colmap_c2w_list.append(aligned_colmap_c2w)
        aligned_c2w_list= [] 
        for i in range(len(cams2world)):
            cam2world = cams2world[i]
            # pose_align_transform = np.eye(4)
            # pose_align_transform[:3, :3] = r_a
            # pose_align_transform[:3, 3] = t_a

            # cam2world[:3,3] = cam2world[:3,3] * s
            # aligned_c2w = pose_align_transform@cam2world
            aligned_c2w = align_transform@cam2world
            # aligned_colmap_c2w[:3,3] = aligned_colmap_c2w[:3,3] * s
            aligned_c2w_list.append(aligned_c2w)
        cams2world = np.stack(aligned_c2w_list)

        scene.apply_transform(align_transform)
        if draw_cameras:
            # add each camera
            for i, pose_c2w in enumerate(aligned_c2w_list):
                if isinstance(cam_color, list):
                    camera_edge_color = cam_color[i]
                else:
                    camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
                pose_c2w[:3,:3] /=s
                add_scene_cam(scene, pose_c2w, camera_edge_color,
                            None if transparent_cams else imgs[i], focals[i],
                            imsize=imgs[i].shape[1::-1], screen_width=cam_size*s)
        
            # add each colmap camera
            for i, pose_c2w in enumerate(colmap_c2w_poses):
                if isinstance(cam_color, list):
                    camera_edge_color = cam_color[i]
                else:
                    camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
                add_scene_cam(scene, pose_c2w, camera_edge_color,
                            None if transparent_cams else imgs[i], focals[i],
                            imsize=imgs[i].shape[1::-1], screen_width=cam_size*s)
        
    else:
        if draw_cameras:
            # add each camera
            for i, pose_c2w in enumerate(cams2world):
                if isinstance(cam_color, list):
                    camera_edge_color = cam_color[i]
                else:
                    camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
                add_scene_cam(scene, pose_c2w, camera_edge_color,
                            None if transparent_cams else imgs[i], focals[i],
                            imsize=imgs[i].shape[1::-1], screen_width=cam_size)
     

    if 0:#transform_to_opengl: #???
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
        # scene.apply_transform(np.linalg.inv(colmap_c2w_poses[0] @ OPENGL @ rot))

    # outfile = os.path.join(outdir, 'scene.glb')
    # outfile = os.path.join(outdir, 'scene.obj')
    outfile = os.path.join(outdir, export_name)

    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile, scene

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, colmap_c2w_poses=None, draw_cameras=True, export_name='scene.obj'):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent, 
                                        colmap_c2w_poses=colmap_c2w_poses, draw_cameras=draw_cameras, export_name=export_name)

def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid, display=False, 
                            cam_poses_c2w=None, im_focals=None, im_pp=None
                            ):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    # scene = global_aligner(output, device=device, mode=mode, verbose=not silent, m_poses=cam_poses_c2w, im_focals=im_focals, im_pp=im_pp, )
    if cam_poses_c2w is not None:
        scene.preset_pose(cam_poses_c2w)
    if im_focals is not None:
        scene.preset_focal(im_focals)

    if im_pp is not None:
        scene.preset_principal_point(im_pp)

    lr = 0.01
    niter = 300

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        # loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        loss = scene.compute_global_alignment(init='known_poses', niter=niter, schedule=schedule, lr=lr)

    outfile, trimesh_scene = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                                     clean_depth, transparent_cams, cam_size, 

                                                     )
    
    if display:
        # Display the scene using trimesh
        trimesh_scene.show()

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d/depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs

def main_demo(tmpdirname, model, device, image_size, input_folder, silent=False):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size, as_pointcloud=False)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent, as_pointcloud=False)
    
    # List all files in the input folder
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    input_files.sort()

    # Filter only image files (assuming common image extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    input_files = [f for f in input_files if os.path.splitext(f)[1].lower() in image_extensions]

    # Check if any image files were found
    if not input_files:
        print("No image files found in the specified folder.")
        return

    # Reconstruction options (same as before)
    schedule = 'linear'
    niter = 'niter'
    scenegraph_type = 'complete'
    winsize = 1 
    refid = 1 
    min_conf_thr = 0.3
    cam_size = 0.05
    as_pointcloud = 1
    mask_sky = 0
    clean_depth = 0
    transparent_cams = 1

    scene, outfile, imgs = recon_fun(input_files, schedule, niter, min_conf_thr, as_pointcloud,
                                     mask_sky, clean_depth, transparent_cams, cam_size,
                                     scenegraph_type, winsize, refid,
                                     
                                     )
    print(f"3D model saved to: {outfile}")




def _read_colmap_pose(extrinsic_path, intrinsic_path, n_views=-1,llffhold=8): 
    from dust3r.utils.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, focal2fov, read_extrinsics_text, read_intrinsics_text
    # cam_extrinsics = read_extrinsics_binary(extrinsic_path)
    # cam_intrinsics = read_intrinsics_binary(intrinsic_path)
    cam_extrinsics = read_extrinsics_text(extrinsic_path)
    cam_intrinsics = read_intrinsics_text(intrinsic_path)


    cameras = []
    for idx, key in enumerate(cam_extrinsics):
        # sys.stdout.write('\r')
        # # the exact output you're looking for:
        # sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec)) # DY: why transpose?
        R = qvec2rotmat(extr.qvec) 
        T = np.array(extr.tvec)
        
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        cameras.append({
                        "name": extr.name,
                        "R": R,
                        "T": T,
                        "focal_length_x": intr.params[0], 
                        "focal_length_y": intr.params[1], 
                        "height": intr.height,
                        "width": intr.width,
                        "FovX": FovX,
                        "FovY": FovY,}
                        )

    cameras.sort(key=lambda x: x["name"])


    train_cam_infos = [c for idx, c in enumerate(cameras) if idx % llffhold != 0]
    test_cam_infos = [c for idx, c in enumerate(cameras) if idx % llffhold == 0]

    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views
    
    return train_cam_infos, test_cam_infos

def reconstruct(tmpdirname, model, device, image_size, input_files, silent=False, colmap_c2w_poses=None, intrinsics=None):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    
    # List all files in the input folder
    # input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    input_files.sort()

    # Filter only image files (assuming common image extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    input_files = [f for f in input_files if os.path.splitext(f)[1].lower() in image_extensions]

    # Check if any image files were found
    if not input_files:
        print("No image files found in the specified folder.")
        return

    # Reconstruction options (same as before)
    schedule = 'linear'
    niter = 'niter'
    scenegraph_type = 'complete'
    winsize = 1 
    refid = 1 
    min_conf_thr = 0.3
    cam_size = 0.05
    as_pointcloud = True #1
    mask_sky = 0
    clean_depth = 0
    transparent_cams = 1

    
    if colmap_c2w_poses is not None:
        cam_poses_c2w, im_focals, im_pp = [], [], []
        for i in range(len(colmap_c2w_poses)):
            assert colmap_c2w_poses[i].shape == (4,4)
            cam_poses_c2w.append(colmap_c2w_poses[i])
            # assert intrinsics[i][0,0] == intrinsics[i][1,1]
            assert np.abs(intrinsics[i][0,0] - intrinsics[i][1,1])<2
            # im_focals.append( 20* np.log(intrinsics[0,0]) )
            im_focals.append(intrinsics[i][0,0])
            im_pp.append(intrinsics[i][:2,2])
    else:
        cam_poses_c2w, im_focals, im_pp = None, None, None

    scene, outfile, imgs = recon_fun(input_files, schedule, niter, min_conf_thr, as_pointcloud,
                                     mask_sky, clean_depth, transparent_cams, cam_size,
                                     scenegraph_type, winsize, refid, cam_poses_c2w=cam_poses_c2w, im_focals=im_focals, im_pp=im_pp, )

    model_from_scene_fun(scene, min_conf_thr=min_conf_thr, as_pointcloud=as_pointcloud, mask_sky=mask_sky,
                            clean_depth=clean_depth, transparent_cams=transparent_cams, cam_size=cam_size, 
                            colmap_c2w_poses=colmap_c2w_poses, draw_cameras=False, export_name='scene.obj'
                            # colmap_c2w_poses=colmap_c2w_poses, draw_cameras=True, export_name='scene.obj'
                            ) 
    print(f"3D model saved to: {outfile}")

    return scene 


def get_pcd_from_sparse_view():
    parser = get_args_parser()
    args = parser.parse_args()
    # Hardcoded input folder
    # input_folder = "/home/user/dust3r/datasets/test"

    for scene in [
        # "03f5c560f5725ad6ca55fd7e6c0af4c4c7a7ca94c444a584f2a9f316d3b35ea2",
        # "0850228cdbf7df721a10d73003db4b8d9d83e17c480b79a6d5d643eff6c8c163",
        # "0a78c25f77c1ba1d1a3f07c18c9735ae1254a9a71290734b8836eefbefaadbc7",
        # "21a66d555d641e7e053718ba492a6e4a85170b4d8cb640566d6cc618a09ea831",
        "25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b",
        # "51a802f3dc0268da35ad944e92cc7266fef00680816eb30d5847d5845b3e867a",
        # "6a4a6c547d4a904697b5be22ce7fbacbeeca85de99badf7f2715aee3fd471a66",
        # "6ed1058f96df97f1c8175739843cf0f272ce0c60c5727dbb010a3a0fac3ef10d",
        # "87c8b2841c276f00d10c53c32ffe628fb26fa3d2cd3ab7bb577ff25d31ee5dbd",
        # "8c508b5b414fdcbbd5387f2dce55c3cdb2200d8ed2038f206126098955cdcdc7",
        # "97f72cff0be96647eeb2fe17ac49752c739af5d1cda656b52e83917a4b2bc17d",
        # "9daa05c4182bb2ea065d280d4f510929d8e9c6d6e18a0782031c7c805cb822ec",
        # "9e4da70fe0be5d28ea7b375291bbf5523246345d807aa47d5208c6e6c2f5694c",
        # "b12e9c2b9b01c1f32aa55567dde42cb831ff6647ee469376677ee890ae36864e",  # has some issues with pose alignment under 3 views   
        # "b7d192fa408f7a8af4a39ab0d8ec3fa6756b614a85c662da9ebf07f57d0e2290",
        # "bd47fd2bd339b8b286470aa40673d829ab646fb92dfc6172e70a9ee966904135",
        # "e7b02299fb49ef1d1ed2a0bb21b014e99e0c2d8aa24d7415085735d653200c95",
        # "ea80b110d4c6ccbb3fd89c6196eb9097e47007d4684f275229634b83a9ac1697",
        # "ef229e255aee1a17530135a934f5fdc4e00aec8b0771cfd0dc183581e508da69",
        # "efdf19ca82bba7bccc73f64273405d077abd61dd2f5339a0a642bc75d7d900ec"
    ]:

        # input_folder = f"/home/yxumich/datadisk/GS_Enhancer/DL3DV/test_split/25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b/images_4/"
        # extrinsic_path = f"/home/yxumich/datadisk/GS_Enhancer/DL3DV/test_split/25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b/colmap/sparse/images.bin"
        # intrinsic_path = f"/home/yxumich/datadisk/GS_Enhancer/DL3DV/test_split/25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b/colmap/sparse/cameras.bin" 
        input_folder = f"/home/yxumich/datadisk/GS_Enhancer/DL3DV/test_split/{scene}/images_4/"
        # extrinsic_path = f"/home/yxumich/datadisk/GS_Enhancer/DL3DV/test_split/{scene}/colmap/sparse/images.bin"
        # intrinsic_path = f"/home/yxumich/datadisk/GS_Enhancer/DL3DV/test_split/{scene}/colmap/sparse/cameras.bin" 
        extrinsic_path = f"/home/yxumich/datadisk/GS_Enhancer/DL3DV/test_split/{scene}/colmap/model/images.txt"
        intrinsic_path = f"/home/yxumich/datadisk/GS_Enhancer/DL3DV/test_split/{scene}/colmap/model/cameras.txt" 


        for n_views in [3, 6, 9]:
            if args.tmp_dir is not None:
                tmp_path = args.tmp_dir
                os.makedirs(tmp_path, exist_ok=True)
                tempfile.tempdir = tmp_path

            if args.server_name is not None:
                server_name = args.server_name
            else:
                server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

            if args.weights is not None:
                weights_path = args.weights
            else:
                weights_path = "naver/" + args.model_name
            model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

            output_dir = f"output/{scene}/{n_views}views" 
            os.makedirs(output_dir, exist_ok=True)
            train_cam_infos, test_cam_infos = _read_colmap_pose(extrinsic_path=extrinsic_path, intrinsic_path=intrinsic_path, n_views=n_views, llffhold=8)
            # import pdb; pdb.set_trace()
            input_files = [ os.path.join(input_folder, cam_info["name"]) for cam_info in train_cam_infos ]

            colmap_c2w_list = []
            intrinsics_list = []
            for cam_info in train_cam_infos:  
                world2cam = np.eye(4)
                world2cam[:3,:3] = cam_info["R"]
                world2cam[:3,3] = cam_info["T"]
                colmap_c2w_list.append(np.linalg.inv(world2cam) )

                intrinsics = np.array([
                    [cam_info["focal_length_x"], 0, cam_info["width"]/2],
                    [0, cam_info["focal_length_y"], cam_info["height"]/2],
                    [0, 0, 1]
                ]) 
                intrinsics[:2] = intrinsics[:2] * 512/cam_info["width"]
                intrinsics_list.append(intrinsics)
            colmap_c2w_list = np.stack(colmap_c2w_list)
            intrinsics_list = np.stack(intrinsics_list)

            reconstruct(output_dir, model, args.device, args.image_size, input_files, silent=args.silent, colmap_c2w_poses=colmap_c2w_list, intrinsics=intrinsics_list)


def post_process_pcd():
    import open3d as o3d
    for scene in [
        # "03f5c560f5725ad6ca55fd7e6c0af4c4c7a7ca94c444a584f2a9f316d3b35ea2",
        # "0850228cdbf7df721a10d73003db4b8d9d83e17c480b79a6d5d643eff6c8c163",
        # "0a78c25f77c1ba1d1a3f07c18c9735ae1254a9a71290734b8836eefbefaadbc7",
        # "21a66d555d641e7e053718ba492a6e4a85170b4d8cb640566d6cc618a09ea831",
        "25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b",
        # "51a802f3dc0268da35ad944e92cc7266fef00680816eb30d5847d5845b3e867a",
        # "6a4a6c547d4a904697b5be22ce7fbacbeeca85de99badf7f2715aee3fd471a66",
        # "6ed1058f96df97f1c8175739843cf0f272ce0c60c5727dbb010a3a0fac3ef10d",
        # "87c8b2841c276f00d10c53c32ffe628fb26fa3d2cd3ab7bb577ff25d31ee5dbd",
        # "8c508b5b414fdcbbd5387f2dce55c3cdb2200d8ed2038f206126098955cdcdc7",
        # "97f72cff0be96647eeb2fe17ac49752c739af5d1cda656b52e83917a4b2bc17d",
        # "9daa05c4182bb2ea065d280d4f510929d8e9c6d6e18a0782031c7c805cb822ec",
        # "9e4da70fe0be5d28ea7b375291bbf5523246345d807aa47d5208c6e6c2f5694c",
        # "b12e9c2b9b01c1f32aa55567dde42cb831ff6647ee469376677ee890ae36864e",  # has some issues with pose alignment under 3 views   
        # "b7d192fa408f7a8af4a39ab0d8ec3fa6756b614a85c662da9ebf07f57d0e2290",
        # "bd47fd2bd339b8b286470aa40673d829ab646fb92dfc6172e70a9ee966904135",
        # "e7b02299fb49ef1d1ed2a0bb21b014e99e0c2d8aa24d7415085735d653200c95",
        # "ea80b110d4c6ccbb3fd89c6196eb9097e47007d4684f275229634b83a9ac1697",
        # "ef229e255aee1a17530135a934f5fdc4e00aec8b0771cfd0dc183581e508da69",
        # "efdf19ca82bba7bccc73f64273405d077abd61dd2f5339a0a642bc75d7d900ec"
        ]:

        for n_views in [3, 6, 9]:
            input_dir = f"/home/yxumich/Projects/Github/dust3r/output/{scene}/{n_views}views/" 
            dust3r_pcd = trimesh.load(f"{input_dir}/scene.obj")

            # plydata = PlyData.read(f"{input_dir}/scene.obj")
            # vertices = plydata['vertex']
            # import pdb; pdb.set_trace()
            # pcd = o3d.io.read_point_cloud()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dust3r_pcd.vertices)
            pcd.colors = o3d.utility.Vector3dVector(dust3r_pcd.colors[:, :3]/255.0)   

            every_k_points = len(pcd.points)//100000
            # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
            uni_down_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
            cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=3.0)

            inlier_cloud = uni_down_pcd.select_by_index(ind)

            # o3d.visualization.draw_geometries([pcd])
            # o3d.visualization.draw_geometries([inlier_cloud])
            # o3d.io.write_point_cloud(f"/home/yxumich/Projects/Github/dust3r/output/{scene}/{n_views}views/scene.ply", inlier_cloud)
            # o3d.io.write_point_cloud(f"{input_dir}/down_sampled.ply", inlier_cloud)
            o3d.io.write_point_cloud(f"{input_dir}/down_sampled_colored.ply", inlier_cloud)


def test():
    import open3d as o3d 
    input_dir = f"/home/yxumich/Downloads/" 
    dust3r_pcd = trimesh.load(f"{input_dir}/dense_views_cyc0.obj")

    # import pdb; pdb.set_trace()
    # pcd = o3d.io.read_point_cloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dust3r_pcd.vertices)
    pcd.colors = o3d.utility.Vector3dVector(dust3r_pcd.colors[:, :3]/255.0)   

    every_k_points = len(pcd.points)//100000
    # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    # uni_down_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
    uni_down_pcd = pcd
    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=100,
                                            std_ratio=1.0)

    inlier_cloud = uni_down_pcd.select_by_index(ind)
    voxel_down_pcd = inlier_cloud.voxel_down_sample(voxel_size=(dust3r_pcd.vertices.max(axis=0)- dust3r_pcd.vertices.min(axis=0)).max()/1000  )

    # o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([inlier_cloud])
    o3d.visualization.draw_geometries([voxel_down_pcd])
    # o3d.io.write_point_cloud(f"/home/yxumich/Projects/Github/dust3r/output/{scene}/{n_views}views/scene.ply", inlier_cloud)
    # o3d.io.write_point_cloud(f"{input_dir}/down_sampled.ply", inlier_cloud)
    o3d.io.write_point_cloud(f"{input_dir}/down_sampled_colored.ply", inlier_cloud)


if __name__ == '__main__':
    get_pcd_from_sparse_view()
    # post_process_pcd()
    # test()
