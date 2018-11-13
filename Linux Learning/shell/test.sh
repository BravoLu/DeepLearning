#!/bin/sh
# -o or  
# -a and 
# -ne not equal 

#数组
array=(1 2 3 4 5)
array2=(aa bb cc dd ee)
value=${array[3]}
echo $value
value2=${array2[3]}
echo $value2
length=${#array[*]} #返回长度
echo $length

echo "hello world" > a.txt  #重定向到文件
echo `date`  #输出当前系统时间

#判断语句
#赋值无空格，其他运算符必须空格
a=10
b=20
if [ $a == $b ]
then 
	echo "a is equal to b"
elif [ $a -gt $b ]
then 
	echo "a is greater than b"
elif [ $a -lt $b ]
then
	echo "a is less than b"
fi

#字符串
#-z 字符串长度为0返回true
#-n 字符串长度不为0返回true



if [ -d a.txt ]
then
	echo "yes"
fi

#for 循环
for FILE in ./*
do 
	echo $FILE
done

test(){
	aNum=3
	anotherNum=5
	return $(($aNum + $anotherNum))
}
test 
result=$?
echo "test result is $result"

#while 循环
COUNTER=0
while [ $COUNTER -lt 5 ]
do 
	COUNTER=`expr $COUNTER + 1`
	echo $COUNTER
done

echo '请输入。。。'
echo 'ctrl + d 即可停止该程序'
while read FILM
do
	echo "Yeah！ great film the $FILM"
done

#跳出循环
#break
#break n
#continue

tes(){
	echo $1 #接受第一个参数
	echo $2 #接受第二个参数
	echo $# #接受参数的个数
	echo $* #接收到所有参数
}

$echo result > file  #追加写
$echo result >> file #覆盖写
echo input < file  #获取输入流