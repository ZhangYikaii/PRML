# a = "你好! "
# print(a)
# b = "今天天气不错!"
# print(a + b)
#
# words = "Xiao Ming Love "
# sth = [
#     "China",
#     "chou doufu",
#     "hecha",
#     "看风景",
#     "电脑",
#     "Python",
#     "跳舞"
# ]
#

a = "你好! "
print(a)
# # i 将循环遍历 sth 里面的所有东西.
# for i in sth:
#     print(words + i)
#
#
# # 让听话的计算机 计算 1 ~ 20 的和:
# ans = 0
# for i in range(21):
#     ans = ans + i
# print(ans)

'''
Xiao Ming love China, chou doufu, hecha, 看风景, 电脑, Python, 跳舞
'''
# # 输出你好10次:
# print("你好\n" * 10)
# for i in range(1, 18879878789789):
#     print("你好")
# print(words + sth[0] + sth[1] + sth[2] + sth[3] + sth[4] + sth[5] + sth[6])
# asr = "" # 告诉计算机 给我一块地方, 它叫asr
# for i in sth:
#     asr = asr + i + ", "
#
# asr = words + asr
# print(asr)

# print(("Hip\n" * 2 + "Hooray!\n") * 3)
# 让 i 在 [0~2]的范围 里面遍历:
# for i in range(2):
#     print("Hip")

'''
我的名字是啥? 正确答案是 小明.
# 只有他答对了, 才可以给他看密码: 123456
'''
a = ""
while a != "小明":
    if a == "是人吗":
        print("你是人吗!!!!")
    else:
        print("你刚刚的回答是: ", a, "它不正确!")
    a = input("请输入答案: ")
print("密码是: 123456")

# 当你的回答不正确的时候:
#      我再问你一遍: 我的名字是啥?
# ars = 1
# while ars < 56756765657576576576564456465:
#     print(ars)
#     ars = ars + 1