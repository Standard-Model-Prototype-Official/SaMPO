# Assignment #2: 编程练习

Updated 0953 GMT+8 Feb 24, 2024

2024 spring, Complied by ==罗景轩，地空==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2 22621.2283 

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：

创建类。一道练习题。

##### 代码

```python
#2300012610ljx
import math

class fraction:
    def __init__(self, top, bottom):
        self.n = top
        self.d = bottom
    
    def show(self):
        print(str(self.n)+'/'+str(self.d))
        return

    def __add__(self, frac):
        nn = self.n * frac.d + self.d * frac.n
        nd = self.d * frac.d
        cm = math.gcd(nn, nd)
        return fraction(nn//cm, nd//cm)
    
a, b, c, d = map(int, input().split())
x, y = fraction(a, b), fraction(c, d)
fraction.show(x + y)
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240303194041338.png" alt="image-20240303194041338" style="zoom:67%;" />



### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



思路：

dp。值得注意的是最后输出`print("%.1f"% VALUE)`

##### 代码

```python
#罗景轩2300012610
n, w = map(int, input().split())
TABLE=[[0]*3 for i in range(n)]

for i in range(n):
    a, b = map(int, input().split())
    TABLE[i]=[a, b, a / b]
TABLE = sorted(TABLE, key=lambda x: x[2], reverse=True)
WEIGHT, VALUE=0, 0.0
for i in range(n):
     if WEIGHT + TABLE[i][1] <= w:
         WEIGHT += TABLE[i][1]
         VALUE += TABLE[i][0]
     else:
         VALUE += (w - WEIGHT)*TABLE[i][2]
         break
print("%.1f" % VALUE)
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240304144943739.png" alt="image-20240304144943739" style="zoom:67%;" />



### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



思路：

排序然后依次减，打得死就alive。

值得注意的是`Skill=sorted(Skill, key= lambda x:(x[0],-x[1]))`的写法

##### 代码

```python
#罗景轩2300012610
nCases = int(input())
for _ in range(nCases):
    Skill=[]
    n, m, b = map(int, input().split())
    count=0
    for _ in range(n):
        Skill.append(tuple(map(int,input().split())))
    Skill=sorted(Skill, key= lambda x:(x[0],-x[1]))
    time=Skill[0][0]
    for i in range(n):
        if time == Skill[i][0]:
            if count <m:
                b -= Skill[i][1]
                count += 1
        else:
            time = Skill[i][0]
            count = 1
            b -= Skill[i][1]
        if b <= 0:
            print(time)
            break
    else:
        print("alive")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240304145219079.png" alt="image-20240304145219079" style="zoom:67%;" />



### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B



思路：

打表、查表。上学期用`in list`判断超时，这次改用`set`。

这里直接导出`is_prime`的真值表，并用两行代码判断是否为完全平方数。

##### 代码

```python
import math
def euler_sieve(n):
    is_prime = [False, False] + [True] * (n - 1)
    
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False                
    return is_prime

P=euler_sieve(1000000)
#print(P)

N=int(input())
List=list(map(int, input().split()))
for i in List:
    sqr = int(math.sqrt(i))
    if P[sqr] and sqr**2 == i:
#   if math.sqrt(i) in P:
        print("YES")
    else:
        print("NO")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240304151009697.png" alt="image-20240304151009697" style="zoom:67%;" />



### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A



思路：

dp.两边倒着来（非常关键）

##### 代码

```python
for _ in range(int(input())):
    n, x = map(int, input().split())
    array = input().split()
    ans = -1
    summ = 0
    for i in range(n):
        summ += int(array[i])
        if summ % x:
            ans = max(ans, max(n - i - 1, i + 1))
    print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==



![image-20240304163723227](C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240304163723227.png)

### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



思路：

没什么好说的，每个过程都实现一遍就好了

##### 代码

```python
#2300012610ljx
import math

P=[False]+[False]+[True]*9999
p=2
while p <= 100:
    if P[p]:
        for i in range(p*2,10001,p):
            P[i]=False
    p+=1
    
def score(L):
    sco = 0
    l=len(L)
    for i in range(l):
        R = int(math.sqrt(L[i]))
        if P[R] and R**2 == L[i]:
            sco += L[i]
    return(sco/l)

m, n = map(int, input().split())
for _ in range(m):
    S = score(list(map(int, input().split())))
    if S==0:
        print(0)
    else:
        print("%.2f"%S)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240304163922239.png" alt="image-20240304163922239" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

本次亮点：class的写法；一些语法小技巧。

类的写法挺数学的，自己可以定义对象，然后定义函数之类的。感觉每一本数学书第一章都是这样，非常亲切，跟回家了一样。题目的分数求和是一个很好的练习。写多了以后代码会非常规整、优美，强迫症福利。模板：

```python
class 'name of the class of certain objects':
    def __init__(self, 'attributes of the object, ...'):
        self.'notation 1' = 'attribute 1'
        self.'notation 2' = 'attribute 2'
        ...
    
    def 'function 1'(self):
        ...
        return 'result'

    def 'function 2'(self):
        ...
        return 'result'
    
    ...
   
#when using functions in class, use the code"name.func()"
```



语法方面：`list = sorted(list, key = lambda x: ……………)`对应的lambda函数可以有多个值，排序的时候会依先后顺序排序，就比如高考分数同分的时候先比数学，这是我们比较熟悉的例子，在python里大概可以这么写：`排名 = sorted(排名, key = lambda x: (x.总分, x.数学成绩, …………))`。

输出真值表的欧拉筛：

```python
def euler_sieve(n):
    is_prime = [False, False] + [True] * (n - 1)
    
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False                
    return is_prime

```

in set ~list[i]>>in list.TLE了可以这么改代码



