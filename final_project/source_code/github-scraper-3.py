from os import write
from github import Github
import requests
from pprint import pprint
import json
import os
import urllib


# Github Enterprise with custom hostname
#g = Github(base_url="https://api.github.com/users", login_or_token="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")
#print(g.get_users)

git_token = "Bearer " + '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'

page_size = 100
url = "https://api.github.com/users?simple=yes&per_page=100&page=1"
image_base = "https://github.com/"
os.chdir("/Users/tanzil/Data/ENSF619.2/Face_Gender_Data/github-profiles")


try:
    urllib.request.urlretrieve("https://github.com/dwabn.png", "930_dwabn.png")
except:
    print("Could not access url", "https://github.com/dwabn.png")
    
print("Downloading will start")
res=requests.get(url,headers={"Authorization": git_token})
print(res.headers, res.content)
users=res.json()
last_downloaded_page = 8
ii = 0
while 'next' in res.links.keys():
  
  res=requests.get(res.links['next']['url'],headers={"Authorization": git_token})
  users.extend(res.json())
  pprint(users)
  
  for jj in range(0,page_size):
    print(ii, jj)
    if ii < last_downloaded_page:
        print("Already donwloaded page:", ii)
        continue
  
    user = users[ii*page_size+jj]
    login = user["login"]
    id = user["id"]
    print(login, id)
    local_filename = str(id)+"_"+login+"_"+".png"
    imageurl = image_base + login + ".png"
    print(imageurl)
    print(local_filename)
    try:
        urllib.request.urlretrieve(imageurl,local_filename)
    except:
        print("Could not access url", imageurl)
  ii = ii+1
  print("Finished Page Number", ii)
  
  #if ii > 0:
  #  break
