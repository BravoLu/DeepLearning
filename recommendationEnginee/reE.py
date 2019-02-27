# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-23 10:35:57
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-23 10:58:42
import pandas as pd 
import numpy as np 
import requests
import json

myun = "BravoLu"
mypw = "39ab6a64a361d92db67fd5936ffb23e07fb3584a"

my_starred_repos = []
def get_starred_by_me():
	resp_list = []
	last_resp = ''
	first_url_to_get = 'https://api.github.com/user/starred'
	first_url_resp   = requests.get(first_url_to_get, auth=(myun,mypw))
	last_resp = first_url_resp
	resp_list.append(json.loads(first_url_resp.text))

	while last_resp.links.get('next'):
		next_url_to_get = last_resp.links['next']['url']
		next_url_resp   = requests.get(next_url_to_get, auth=(myun,mypw))
		last_resp = next_url_resp
		resp_list.append(json.loads(next_url_resp.text))


	for i in resp_list:
		for j in i:
			msr = j['html_url']
			my_starred_repos.append(msr)


get_starred_by_me()
# for i in my_starred_repos:
# 	print(i)

my_starred_users = []
for ln in my_starred_repos:
	right_split = ln.split('.com/')[1]
	starred_usr = right_split.split('/')[0]
	my_starred_users.append(starred_usr)

