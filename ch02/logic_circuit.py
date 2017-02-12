import numpy as np
from collections import OrderedDict

class LogicGate:

    def __init__(self, w, b):
        self._W = w
        self._b = b

    def forward(self, x):
        s = np.dot(x, self._W) + self._b
        y = self.activate_func(s)
        return y

    def activate_func(self,s):
        if s < 0 :
            return 0
        else:
            return 1

class And(LogicGate):

    def __init__(self):
        w = np.array([0.5, 0.5])
        b = -0.7
        super().__init__(w, b)

class Nand(LogicGate):

    def __init__(self):
        w = np.array([-0.5, -0.5])
        b = 0.7
        super().__init__(w, b)

class Or(LogicGate):

    def __init__(self):
        w = np.array([0.5, 0.5])
        b = -0.2
        super().__init__(w, b)


class Xor:

    def __init__(self):
        pass

    def forward(self, x):

        s1 = Nand().forward(x)
        s2 = Or().forward(x)
        y  = And().forward(np.array([s1, s2]))
        return y





if __name__ == '__main__':
    # 論理値入力
    LOGIC_INPUTS = [np.array([0,0])
                    ,np.array([0,1])
                    ,np.array([1,0])
                    ,np.array([1,1])]

    logics = OrderedDict()
    logics['And'] = And()
    logics['Nand'] = Nand()
    logics['Or'] = Or()
    logics['Xor'] = Xor()

    for key in logics:
        print(key + "--------")
        for x in LOGIC_INPUTS:
            logic = logics[key]
            y = logic.forward(x)
            print(str(x) + "->" + str(y))


