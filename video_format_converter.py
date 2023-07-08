import ffmpy
import easygui
import os

inputs = []
index = 0
errors = []
input_files = None

path = easygui.diropenbox(title='Attention', msg='Please choose the folder.')
print('input_path : ', path, '\n')

for root ,dirs, files in os.walk(path):
    break
if len(files) == 0:
    easygui.msgbox(title='Attetion', msg='This folder seems to be empty.')
elif len(files) == 1:
    choice = easygui.buttonbox(title='Attention', msg='Do you need to split '+files[0]+' ?', choices=['Yes', 'No'])
    if choice == 'Yes':
        input_files = files
else:
    input_files = easygui.multchoicebox(title='Attention', msg='Please choose the videos.', choices=files)
print('video_files : ', input_files, '\n')
if input_files:
    for file in input_files:
        inputs.append(path+'\\'+file)

if inputs:
    outputs = easygui.diropenbox(title='Attention', msg='Please choose the save path.') + '\\'
    print('output_path : ', outputs, '\n')

    print('Processing...', '\n')

    for video in inputs:
        try:
            source_file = video
            sink_file = outputs + files[index].split('.')[0] + '.mp4'
            ff = ffmpy.FFmpeg(executable='D:/PROGRAM/Python/Lib/site-packages/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe',
                              inputs = {source_file: None},
                              outputs = {sink_file: None}).run()
        except:
            errors.append(sink_file)
        index += 1

    if errors:
        easygui.msgbox(title='Attetion', msg=str(errors)+'\n'+"is/are exiting and others' formats are converted.")
    else:
        easygui.msgbox(title='Attetion', msg='All video formats are converted.')