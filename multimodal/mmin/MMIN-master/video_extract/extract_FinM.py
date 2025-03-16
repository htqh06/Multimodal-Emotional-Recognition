import ffmpeg
import os


def extract_video_segment(video_path, start_time, end_time, output_folder, identifier):
    """使用 ffmpeg 裁剪视频的左侧部分，并重新编码。"""
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    crop_width = width // 2  # 计算裁剪宽度为视频宽度的一半

    output_path = os.path.join(output_folder, f"{identifier}.mp4")  # 使用标识符作为输出文件名


    try:
        input_video = ffmpeg.input(video_path, ss=start_time, to=end_time)
        # 裁剪视频并保留音频流
        video = input_video.video.filter('crop', w=crop_width, h=height, x=crop_width, y=0)
        audio = input_video.audio
        (
            ffmpeg
            .concat(video, audio, v=1, a=1)
            .output(output_path, vcodec='libx264', acodec='aac')
            .run()
        )
        print(f"Processed video saved to {output_path}")
    except ffmpeg.Error as e:
        print(f"An error occurred while processing the video {identifier}: {e}")


def process_videos(timestamp_folder, video_folder, output_folder):
    """处理每个视频和对应的时间戳文件，为每个视频创建一个单独的文件夹来存储裁剪的结果。"""
    for timestamp_file in os.listdir(timestamp_folder):
        if timestamp_file.endswith(".lab") and 'M' in timestamp_file:
            session_id = timestamp_file[:-4]
            video_path = os.path.join(video_folder, f"{session_id}.avi")
            session_output_folder = os.path.join(output_folder, session_id)  # 为每个视频创建一个文件夹

            # 确保目标文件夹存在
            if not os.path.exists(session_output_folder):
                os.makedirs(session_output_folder)

            with open(os.path.join(timestamp_folder, timestamp_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3 and parts[2].startswith('Ses'):  # 确保行格式正确且标识符存在
                        start_time, end_time, identifier = parts
                        extract_video_segment(video_path, float(start_time), float(end_time), session_output_folder,
                                              identifier)


if __name__ == "__main__":
    timestamp_folder = 'D:/Graduation_IEMOCAP/IEMOCAP_full_release/Session2/dialog/lab/Ses02_F/'
    video_folder = 'D:/Graduation_IEMOCAP/IEMOCAP_full_release/Session2/dialog/avi/DivX'
    output_folder = 'D:/Graduation_IEMOCAP/IEMOCAP_extract/IEMOCAP/processed/Session2'
    process_videos(timestamp_folder, video_folder, output_folder)