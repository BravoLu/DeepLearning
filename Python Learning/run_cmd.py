import argparse
import subprocess
import os
from PIL import Image
import re
import shutil


parser = argparse.ArgumentParser(description="extract feature from a image")
parser.add_argument('--img', default='', type=str, help='input the image name')
face_detect = './facetrackpro/detect_test ../facetrackpro/model/detect/3.3.5/'
face_align  = './facetrackpro/track_test'
detected_rect_path = 'img_detected_v2'
all_file_dir = 'all_file'
all_file_name_list = []
#test_list = ['_02aaa2.jpg']

def judge_png(file_name):
	if file_name[-3:] == 'jpg' or file_name[-3:] == 'bmp' or file_name[-3:] == 'png':
		return True
	return False



def get_all_file_name():
	path = 'imgs'
	new_path = all_file_dir 
	for root,dirs,files in os.walk(path):
		for i in range(len(files)):
			if judge_png(files[i]):
				all_file_name_list.append(files[i])
			file_path = root + '/' + files[i]
			new_file_path = new_path + '/' + files[i]
			shutil.copyfile(file_path,new_file_path)
	#print(all_file_name_list)

				
	

def run_cmd2str(cmd):
	result_str = ''
	process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	result_f = process.stdout
	error_f  = process.stderr
	errors   = error_f.read()
	if errors: pass
	result_str = result_f.read().strip()
	if result_f:
		result_f.close()
	if error_f:
		error_f.close()
	return result_str
		
def run_cmd2file(cmd):
	fdout = open("rect.log",'a') 
	p = subprocess.Popen(cmd,stdout=fdout,stderr=fdout,shell=True)
	if p.poll():
		return
	p.wait()
	return

# get the number of face from after detection
def get_face_number(input_str):
	face_num = 0
	print(input_str)
	list = input_str.split('\n')
	#print(list)
	if len(list) < 3:
		return 0
	face_num = int(list[3].split()[2])
	return face_num

def get_face_rect(input_str, face_num, img_name):
	list = input_str.split('\n')
	if face_num == 0:
		#print('false')
		return 
	for i in range(face_num):
		tmp_str = list[4+i]
		rect_list = re.findall(r"\d+\.?\d*",tmp_str)
		img = Image.open(img_name)
		left = int(rect_list[2])
		right = int(rect_list[3])
		upper = left + int(rect_list[0])
		lower = right + int(rect_list[0])
		cropped = img.crop((left, right, upper, lower))
		tmp = os.path.split(img_name)[-1]
		FileName = tmp.split('.')[0]
		postfix = tmp.split('.')[-1]
		
		rect_file_name = os.path.join(detected_rect_path, FileName+'_{}.{}'.format(i,'rect'))
		with open(rect_file_name,'w') as f:
			f.write('0 ')
			f.write('0 ')
			f.write(rect_list[0]+' ')
			f.write(rect_list[0])
			
		print('##########{}'.format(FileName))
		save_file_name = FileName + '_{}.{}'.format(i,postfix)
		try:
			cropped.save(os.path.join(detected_rect_path,save_file_name))
		except IOError,e:
			print(e.message)

		print('saving...')
		#cropped.save(os.path.join(detected_rect_path,'{}_'.format(i),img_name))
		#print(rect_list)
		
def get_all_detected_file():
	path = 'all_file'
	for filename in all_file_name_list:
		shell = face_detect + ' {}'.format(os.path.join(path,filename))
		print('shell:{}'.format(shell))
		cmd_out = run_cmd2str(shell)
		face_number = get_face_number(cmd_out)
		get_face_rect(cmd_out,face_number,os.path.join(path,filename))



def main():
	get_all_file_name()
	#args = parser.parse_args()
	get_all_detected_file()
	#img_name = args.img
	#shell = face_detect + ' {}'.format(img_name)
	#cmd_out = run_cmd2str(shell)
	#os.system(shell)
	#run_cmd2file(shell)
	#print(run_cmd2str(shell))
	#print(get_face_number(run_cmd2str(shell)))
	#face_number = get_face_number(run_cmd2str(shell))
	#get_face_rect(cmd_out,face_number, img_name)
		


if __name__ == '__main__':
	main()

