import matplotlib.pylab as plt

def add_instance_method(Klass, method):
    """クラスにインスタンスメソッドを追加する"""
    setattr(Klass, method.__name__, method)


def showActivateWindow():
    """
    plt.show()してもウィンドウが前面に出ないため
    出るようにする
    """
    cfm = plt.get_current_fig_manager()
    cfm.window.activateWindow()
    cfm.window.raise_()
    plt.show()


# add_instance_method()

def plt_show_focus(plt):

    '''
    plt.show()してもウィンドウが前面に出ないため
    出るようにする

    Parameters
    ----------
    plt

    Returns
    -------

    '''
    cfm = plt.get_current_fig_manager()
    cfm.window.activateWindow()
    cfm.window.raise_()
    plt.show()

