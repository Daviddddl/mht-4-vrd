import functools
import os
import re
import subprocess

import cv2
import numpy as np

"""
sudo add-apt-repository ppa:djcj/hybrid
sudo apt-get update
sudo apt-get install ffmpeg
"""


def extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass
    if num_frames == -1:
        return extract_all_frames(video_file)

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    # print(output)
    extract_frame_paths = sorted([os.path.join('frames', frame)
                                  for frame in os.listdir('frames')])

    res_frames = load_frames(extract_frame_paths)
    # subprocess.call(['rm', '-rf', 'frames'])
    return res_frames, extract_frame_paths


def extract_all_frames(video_file, out_path=None):
    if out_path is None:
        try:
            os.makedirs(os.path.join(os.getcwd(), 'frames/' + video_file[:-4]))
        except OSError:
            pass
        extract_frame_path = os.getcwd() + '/frames/' + video_file[:-4]
    else:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        extract_frame_path = out_path
    os.system('ffmpeg -i ' + video_file + ' ' + extract_frame_path + '/%4d.jpg')

    return extract_frame_path


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


if __name__ == '__main__':
    # for root, dirs, files in os.walk('vis_out/visualization'):
    #     for each_file in files:
    #         print(extract_frames(each_file))

    video_path = '/home/daivd/Desktop/test/test.mp4'
    out_path = '/home/daivd/Desktop/test_out'
    extract_all_frames(video_path, out_path)
