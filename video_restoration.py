import cv2
import os
import glob
import moviepy.editor
import easygui
import torch
import shutil

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from tqdm import tqdm
from threading import Thread



def get_information(input_video):

    videoCapture = cv2.VideoCapture(input_video)
    fps = float(videoCapture.get(cv2.CAP_PROP_FPS))
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) * 4
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 4
    video_name = input_video[:-1-len(input_video.split('.')[-1])].split('\\')[-1]
    images_path = temp_path + video_name

    return video_name, images_path, fps, frames, width, height



def load_model(model_name, gpu_id, face_enhance):

    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    model_path = os.path.join('weights', model_name + '.pth')

    upsampler = RealESRGANer(scale=netscale,
                             model_path=model_path,
                             dni_weight=None,
                             model=model,
                             tile=0,
                             tile_pad=10,
                             pre_pad=0,
                             half=True,
                             gpu_id=gpu_id)

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    else:
        face_enhancer = None

    return upsampler, face_enhancer


def video_split(input_video, temp_path, video_name, images_path, frames):

    global done
    cap = cv2.VideoCapture(input_video)
    folder = os.path.exists(images_path)
    if not folder:
        os.makedirs(images_path)

    for i in tqdm(range(0, frames), desc='Splitting'):
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                savePath = images_path + '\\' + str(i+1).zfill(8) + ".jpg"
                cv2.imencode('.jpg', frame)[1].tofile(savePath)
                done['split'] = i + 1
        else:
            break



def restore(input, output_path, face_enhance, frames, upsampler, face_enhancer):

    global done
    os.makedirs(output_path, exist_ok=True)
    with tqdm(range(0, frames), desc='Restoring') as tbar:
        for i in tbar:
            while True:
                paths = sorted(glob.glob(os.path.join(input, '*')))
                if (paths and i+1 < done['split']) or done['split'] == frames:
                    path = paths[0]
                    imgname, extension = os.path.splitext(os.path.basename(path))
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img_mode = 'RGBA'
                    else:
                        img_mode = None
                    try:
                        if face_enhance:
                            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                        else:
                            output, _ = upsampler.enhance(img, outscale=4)
                    except RuntimeError as error:
                        print('Error', error)
                        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                    else:
                        extension = extension[1:]
                        if img_mode == 'RGBA':  # RGBA images should be saved in png format
                            extension = 'png'
                        save_path = os.path.join(output_path, f'{imgname}.{extension}')
                        cv2.imwrite(save_path, output)
                    os.remove(path)
                    done['restore'] = i + 1
                    break



def integrate(input_path, temp_path, video_name, fps, frames, save_path, width, height):

    global done
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))
    with tqdm(range(0, frames), desc='Integrating') as tbar:
        for i in tbar:
            while True:
                paths = sorted(glob.glob(os.path.join(input_path, '*')))
                if (paths and i+1 < done['restore']) or done['restore'] == frames:
                    path = paths[0]
                    img = cv2.imread(path)     
                    img = cv2.resize(img, (width, height))
                    video.write(img)
                    os.remove(path)
                    break
    video.release()



def audio_split(audio_path, audio_source):

    audio = moviepy.editor.VideoFileClip(audio_source)
    audio.audio.write_audiofile(audio_path)



def audio_integrate(audio_integrated_to, video_name, output_path, audio_path):

    audio = moviepy.editor.AudioFileClip(audio_path)
    video = moviepy.editor.VideoFileClip(audio_integrated_to)
    video = video.set_audio(audio)
    video.write_videofile(output_path + '\\' + video_name + '_complete.mp4')



def video_restore(input_video, output_path, temp_path, model_name, face_enhance, gpu_id, upsampler, face_enhancer):

    video_name, images_path, fps, frames, width, height = get_information(input_video)
    output = images_path + '_restored\\'
    save_path = temp_path + video_name + '_restored' + '.mp4'
    audio_path = temp_path + video_name + '.mp3'
    print('Video ', video_name, ' is being processed')
    audio_split_thread = Thread(target=audio_split, args=(audio_path, input_video))
    video_split_thread = Thread(target=video_split, args=(input_video, temp_path, video_name, images_path, frames))
    restore_thread = Thread(target=restore, args=(images_path, output, face_enhance, frames, upsampler, face_enhancer))
    integrate_thread = Thread(target=integrate, args=(output, temp_path, video_name, fps, frames, save_path, width, height))
    audio_split_thread.start()
    print('Audio split thread start')
    video_split_thread.start()
    print('Video split thread start')
    restore_thread.start()
    print('Restore thread start')
    integrate_thread.start()
    print('Integrate thread start')
    audio_split_thread.join()
    video_split_thread.join()
    restore_thread.join()
    integrate_thread.join()
    audio_integrate(save_path, video_name, output_path, audio_path)

    print('Temp removing...')
    shutil.rmtree(temp_path, ignore_errors=False, onerror=None)
    print('Complete the removement', '\n')



if __name__ == '__main__':

    temp_path = '.\\temp\\'
    gpus_device = []
    gpus_device_id = {}
    inputs = []
    errors = []
    input_files = []
    done = {'split':0, 'restore':0}

    input_path = easygui.diropenbox(title='Attention', msg='Please choose the folder(only mp4 videos are available).')
    print('input_path : ', input_path)

    for root ,dirs, files in os.walk(input_path):
        break
    if len(files) == 0:
        print('This folder seems to be empty.')
    elif len(files) == 1:
        choice = easygui.buttonbox(title='Attention', msg='Do you need to restore '+files[0]+' ?', choices=['Yes', 'No'])
        if choice == 'Yes':
            input_files = files
    else:
        input_files = easygui.multchoicebox(title='Attention', msg='Please choose the videos(only mp4 videos are available).', choices=files)
    print('video_files : ', input_files)

    if input_files:
        for file in input_files:
            inputs.append(input_path + '\\' + file)

        output_path = easygui.diropenbox(title='Attention', msg='Please choose the save folder.')
        print('output_path : ', output_path)

        model_name = easygui.choicebox(title='Attention', msg='Please choose the model.', choices=['RealESRGAN_x4plus', 'realesr-animevideov3'])
        print('model_name : ', model_name)

        face_enhance = easygui.buttonbox(title='Attention', msg='Bo you need to enhance face ?', choices=['Yes', 'No'])
        if face_enhance == 'Yes':
            face_enhance = True
        else:
            face_enhance = False
        print('face_enhance : ', face_enhance)

        if torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            for i in range(0, total_gpus-1):
                gpus_device.append(torch.cuda.get_device_name(0))
                gpus_device_id[torch.cuda.get_device_name(0)] = i
            if total_gpus > 1:
                gpu = easygui.choicebox(title='Attention', msg='Please choose the GPU.', choices=gpus_device)
                gpu_id = gpus_device_id[gpu]
            else:
                gpu_id = 0
        else:
            gpu_id = None
        print('gpu_id : ', gpu_id, '\n')

    upsampler, face_enhancer = load_model(model_name, gpu_id, face_enhance)

    if inputs:
        for video in inputs:
            try:
                video_restore(video, output_path, temp_path, model_name, face_enhance, gpu_id, upsampler, face_enhancer)
            except:
                errors.append(video.split('\\')[-1])
        if errors:
            print(str(errors) + "failed and others are successful")
        else:
            print('All videos are successful')