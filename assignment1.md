# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by ==罗景轩，地空==



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2 22621.2283 

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)



## 1. 题目

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



思路：

打表，查表。以免计算过多

##### 代码

```python
#2300012610罗景轩
T = [0, 1, 1]
for i in range(28):
    T += T[i] + T[i+1] + T[i+2],
    
print(T[int(input())])
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240303163236789.png" alt="image-20240303163236789" style="zoom:67%;" />



### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



思路：

双指针，分别指向目标的字符串与待检验的字符串，依次匹配。

##### 代码

```python
#2300012610ljx
target = 'hello'
i, j = 0, 0
obj = str(input())
while j < len(obj) and i!=5:
    if target[i] == obj[j]:
        i += 1
    j += 1
    
if i == 5:
    print("YES")
else:
    print("NO")

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240303164818383.png" alt="image-20240303164818383" style="zoom:67%;" />



### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



思路：

先全部小写，然后挨个检验是否是元音。

##### 代码

```python
#2300012610ljx
vowels = ['a', 'e', 'i', 'o', 'u', 'y']
obj = str(input()).lower()
ans = ''
for i in obj:
    if i not in vowels:
        ans += "."+i
print(ans)      

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240303175417604.png" alt="image-20240303175417604" style="zoom:67%;" />



### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



思路：

先用筛法找出范围内的素数，再检验

##### 代码

```python
#2300012610ljx
def euler_sieve(n):
    is_prime = [True] * (n + 1)
    primes = []
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
                
    return primes

l = euler_sieve(10000)
n = int(input())
for i in l:
    if n - i in l:
        print(i, n-i)
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240303180024858.png" alt="image-20240303180024858" style="zoom:67%;" />



### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



思路：

比较繁琐的多次判断，逻辑对了就不容易错，但是坑比较多。把上学期的代码翻出来贴上去过了。

##### 代码

```python
#2300012610ljx
polynomial=input()
Degree=0
if polynomial.isdigit():
    print("n^0")
else:
    terms=list(polynomial.split("+"))
    for term in terms:
        coefficient, degree=term.split("n^")
        if coefficient == "" or int(coefficient) != 0:
            if int(degree) > Degree:
                Degree = int(degree)
    print("n^%d"%Degree)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240303185626353.png" alt="image-20240303185626353" style="zoom:67%;" />



### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



思路：

计数，比较，排序，输出

##### 代码

```python
#2300012610ljx
def func(votes):
    count = {}
    for i in votes:
        if i in count:
            count[i] += 1  
        else:
            count[i] = 1
    maxx = max(count.values())
    ans = []
    for a, b in count.items():
       if b == maxx:
           ans.append(a)
    ans.sort(key = lambda x: int(x))
    return ans

votes = input().split()
ans = func(votes)
print(*ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240303201351279.png" alt="image-20240303201351279" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“数算pre每日选做”、CF、LeetCode、洛谷等网站题目。==

这一辑注重python基础，所以很多题目都和上学期的计概课程重复。有幸再次成为闫老师的学生，所以对于交作业的方式之类都比较熟悉，这一方面的烦恼少了很多。

由于第一周事情比较多，第一次作业留到了3.3第二周周天写，比较极限，好在大部分题目上学期做过，找了一下自己的AC代码就过了。通过一些题目，我发现了自己在python基础语法方面存在的一些薄弱。笔记如下：

欧拉筛：

```python
def euler_sieve(n):
    is_prime = [True] * (n + 1)
    primes = []
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
                
    return primes
```

比较不好理解，可以放cheat paper上。

`return`在函数内跳出，`break`跳出循环，`continue`跳出这次循环，继续。

`while`在满足条件后继续，和用于可迭代对象的`for`各有千秋。

`while/for ... else ...`多重判断，不好理解所以尽量别写。

可迭代对象的索引，正数从零开始，倒数从-1开始递减。字符串也是可迭代对象，注意到这点很多题目会很省事。

平时一部分函数需要以`class.func()`的形式使用，实际上是特定类的函数，作为比较，平时写类的时候，也需要这么写类的函数。不懂的时候可以看看实现的代码，既学学类的写法，也可以记一记常见函数的实现以及输入输出的类型。

举例：

```python
dic = {'a':1, 'b':2, 'c':3}
print(dic.keys())
print(dic.values())
print(dic.items())
#输出
#dict_keys(['a', 'b', 'c'])
#dict_values([1, 2, 3])
#dict_items([('a', 1), ('b', 2), ('c', 3)])
```

另外`*list`的写法很好用也很好理解，建议推广。
