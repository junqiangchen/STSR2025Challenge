import numpy as np
import open3d as o3d
import os


def load_stl_pointcloud(stl_path):
    """
    1) 读取 STL 网格；
    2) 法线估计 + Poisson 采样点云。
    """
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    print(int(mesh.vertices.__len__() * 0.5))
    if int(mesh.vertices.__len__() * 0.5) > 50000:
        number_points = 50000
    else:
        number_points = int(mesh.vertices.__len__() * 0.5)
    pcd = mesh.sample_points_poisson_disk(5000)
    return mesh, pcd


def preprocess_pointcloud(pcd, voxel_size=1.0):
    """
    体素下采样 + 法线估计
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down


def refine_registration(source, target, init_trans, voxel_size=1.0):
    """
    基于 ICP 的精配准
    """
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


def task2inference(imagedir, outputdir, isoname):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    # 0. 分割牙齿区域的mask，并从原图中进行提取ROI范围
    iosstlpath = imagedir + "/" + isoname
    cbctstlpath = outputdir + "/cbct.stl"
    # 1. 加载点云
    cbct_mesh, cbct_pcd = load_stl_pointcloud(cbctstlpath)
    intra_mesh, intra_pcd = load_stl_pointcloud(iosstlpath)
    # upper_gt = np.load(
    #     r"F:\MedicalData\2025STSR\download\MICCAI-Challenge-STSR2025-Task2\Train-Labeled\Labels\010\upper_gt.npy")
    # print("Global registration transformation:\n", upper_gt)
    # intra_pcd.transform(upper_gt)
    # o3d.visualization.draw_geometries([
    #     cbct_pcd.paint_uniform_color([0.7, 0.7, 0.7]),
    #     intra_pcd.paint_uniform_color([1, 0, 0])
    # ], window_name="CBCT (gray) vs Intraoral Scan (red)")
    # 2. 预处理（下采样+求法线）
    voxel_size = 1.0  # 根据数据分辨率调整
    src_down = preprocess_pointcloud(intra_pcd, voxel_size)
    tgt_down = preprocess_pointcloud(cbct_pcd, voxel_size)
    # 3. 粗配准,将两个点云的质心对齐
    centroid_src = np.mean(np.asarray(src_down.points), axis=0)
    centroid_tgt = np.mean(np.asarray(tgt_down.points), axis=0)
    init_trans = np.eye(4)
    init_trans[:3, 3] = centroid_tgt - centroid_src
    print("Global registration transformation:\n", init_trans)
    # intra_pcd.transform(init_trans)
    # o3d.visualization.draw_geometries([
    #     cbct_pcd.paint_uniform_color([0.7, 0.7, 0.7]),
    #     intra_pcd.paint_uniform_color([1, 0, 0])
    # ], window_name="CBCT (gray) vs Intraoral Scan (red)")
    # 4. 精配准
    refined = refine_registration(intra_pcd, cbct_pcd, init_trans, voxel_size)
    print("Refined transformation:\n", refined.transformation)
    # 5. 应用变换并可视化
    intra_pcd.transform(refined.transformation)
    # o3d.visualization.draw_geometries([
    #     cbct_pcd.paint_uniform_color([0.7, 0.7, 0.7]),
    #     intra_pcd.paint_uniform_color([1, 0, 0])
    # ], window_name="CBCT (gray) vs Intraoral Scan (red)")
    intra_mesh.transform(refined.transformation)
    cbctstlpath = outputdir + "/" + isoname
    o3d.io.write_triangle_mesh(cbctstlpath, intra_mesh)


if __name__ == "__main__":
    imagedir = r"F:\MedicalData\2025STSR\dataset\task2"
    task2inference(imagedir, imagedir + '/pd', "lower.stl")
    task2inference(imagedir, imagedir + '/pd', "upper.stl")
