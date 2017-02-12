# coding: utf-8
class Man:
    """サンプルクラス"""

    #クラス変数
    num8 = 3

    def __init__(self, name):
        self.name = name
        print("Initilized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

    def hoge(this,age):
        '''selfは別の変数名でも平気'''
        print("this=" + str(this))
        print("name=" + this.name)
        print("age=" + str(age))


    @staticmethod
    def hoge2(str):
        print(str)
        print(Man.num8)

    @classmethod
    def cls_method(cls,str):
        print("str={} Man.num8={}".format(str,cls.num8))

m = Man("David")
m.hello()
m.goodbye()
m.hoge(2)
m.hoge2('hoge2')
Man.hoge2('hoge2')
Man.cls_method('hoge3')