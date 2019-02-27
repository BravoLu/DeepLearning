# nginx
## 正向代理
1. resolver:DNS服务器IP地址
2. listen:主动发起请求的内网服务器端口
3. proxy_pass:代理服务器的协议和地址

## 反向代理
1. server_name:代表客户端向服务器发起请求时输入的域名
2. proxy_pass:代表源服务器的访问地址，也就是真正请求的服务器（localhost+端口号）。

## 透明代理
也叫做简单代理，意思客户端向服务端发起请求时，请求会先到达透明代理服务器，代理服务器再把请求转交给真实的源服务器处理，也就是客户端根本不知道有代理服务器的存在。

## 负载均衡
将服务器接受到的请求按照规则分发的过程，称为负载均衡。负载均衡是反向代理的一种体现。
1. 轮询
```shell
	upstream serverList{
		server 1.2.3.4;
		server 1.2.3.5;
		server 1.2.3.6;
	}
```
2. ip_hash
```shell
	upstream serverList{
		ip_hash
		server 1.2.3.4;
		server 1.2.3.5;
		server 1.2.3.6;
	}
```
3. url_hash
按访问url的hash结果来分配请求，相同的url固定转发到同一个后端服务器处理
```shell
	upstream serverList{
		server 1.2.3.4;
		server 1.2.3.5;
		server 1.2.3.6;
		hash $request_uri;
		hash_method crc32;
	}
```
4. fair:按后端服务器的响应时间来分配请求，响应时间短的有限分配。
```shell
	upstream serverList{
		server 1.2.3.4;
		server 1.2.3.5;
		server 1.2.3.6;
		fair;
	}
```
|服务器后的参数|参数值|
|-|-|
|down|当前服务器暂不参与负载|
|weight|服务器的负载量|
|max_fails|允许请求失败的次数|
|fail_timeout|max_fails次失败后暂停的时间|
|backup|备份机，只有其他所有的非backup机down或忙时才会请求backup机器|

e.g
```
upstream serverList{
	server 1.2.3.4 weight=30;
	server 1.2.3.5 down;
	server 1.2.3.6 backup;
}

server {
	listen 80;
	server_name www.xxx.com;
	root html;
	index index.html index.htm index.php;
	location / {
		proxy_pass http://serverList;
		proxy_redirect off;
		proxy_set_header Host $host;
	}
}
```

## 静态服务器
nginx作为静态服务器,处理前端。
1. root：直接静态项目的绝对路径的根目录。
2. server_name: 静态网站访问的域名地址。
```
server{
	listen 80;
	server_name www.xxx.com;

	client_max_body_size 1024M;
	location / {
		root /var/www/xxx_static;
		index index.html
	}
}
```