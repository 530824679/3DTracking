
from __future__ import print_function
import time
import os.path
import numpy as np
from tracker.tracking import MOT3D
from utils.general import load_list_from_folder, fileparts, mkdir_if_missing


if __name__ == '__main__':
    result_sha = "car_3d_det"
    save_root = './results'

    # KITTI dataset format
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    seq_file_list, num_seq = load_list_from_folder(os.path.join('data/KITTI', result_sha))

    total_time = 0.0
    total_frames = 0

    save_dir = os.path.join(save_root, result_sha)
    mkdir_if_missing(save_dir)
    eval_dir = os.path.join(save_dir, 'data')
    mkdir_if_missing(eval_dir)

    for seq_file in seq_file_list:
        _, seq_name, _ = fileparts(seq_file)

        print("seq:", seq_name)
        print("seq_file:", seq_file)

        mot_tracker = MOT3D()

        # load KITTI detections files
        seq_dets = np.loadtxt(seq_file, delimiter=',')
        eval_file = os.path.join(eval_dir, seq_name + '.txt')
        eval_file = open(eval_file, 'w')
        save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name)
        mkdir_if_missing(save_trk_dir)
        print("Processing %s." % (seq_name))

        for frame in range(int(seq_dets[:, 0].min()), int(seq_dets[:, 0].max()) + 1):
            print(frame)

            save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame)
            save_trk_file = open(save_trk_file, 'w')
            dets = seq_dets[seq_dets[:, 0] == frame, 7:14]

            ori_array = seq_dets[seq_dets[:, 0] == frame, -1].reshape((-1, 1))
            other_array = seq_dets[seq_dets[:, 0] == frame, 1:7]
            additional_info = np.concatenate((ori_array, other_array), axis=1)
            dets_all = {'dets': dets, 'info': additional_info}
            total_frames += 1
            start_time = time.time()
            trackers = mot_tracker.update(dets_all, seq_name, frame)
            cycle_time = time.time() - start_time
            total_time += cycle_time
            for d in trackers:
                bbox3d_tmp = d[0:7]
                id_tmp = d[7]
                ori_tmp = d[8]
                type_tmp = det_id2str[d[9]]
                bbox2d_tmp_trk = d[10:14]
                conf_tmp = d[14]

                str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
                                bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3],
                                bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3],
                                bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6],
                                conf_tmp, id_tmp)
                save_trk_file.write(str_to_srite)

                str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp,
                                type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1],
                                bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], bbox3d_tmp[0], bbox3d_tmp[1],
                                bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6],
                                conf_tmp)
                eval_file.write(str_to_srite)

            save_trk_file.close()

        eval_file.close()

        del mot_tracker

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))