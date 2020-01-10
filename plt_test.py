
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def nine1():
    data = np.arange(10)

     #これで表示

    fig = plt.figure()
    #ax1 = fig.add_subplot(1,1,1)    #2*2個のサブプロットのうち1つ目を選択
    #ax1.cla()
    
    #plt.plot(data)                  #1つ目のサブプロットにdataを表示


    #ax2 = fig.add_subplot(2,2,2)    #2*2個のサブプロットのうち2つ目を選択
   

    #ax3 = fig.add_subplot(1,1,1)    #2*2個のサブプロットのうち3つ目を選択
    ax3 = plt.subplots(1, figsize=(16, 16))[1]

    plt.plot(np.random.randn(50).cumsum(), "k--")
                                    #cumsum():配列内の要素を足し合わせていったものを
                                    # 順次配列に記録していていくもの
                                    # 近似的な積分を行なっているイメージ
                                    #k--:グラフを点線にする


    #_ = ax1.hist(np.random.randn(100), bins = 20, color = "k", alpha = 0.3)
            #hist:ヒストグラムを描画
            #bins:ビン数
            #color:色指定(たぶんkで黒)
            #alpha:色の透明度？(RGBAのA？)

    #ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
        #点を描画　
        #randn(30) :平均0、標準偏差1の標準正規分布を30個作る

    #fig, axes = plt.subplots(2,3)   #2(縦)*3(横)個のサブプロットを生成
                    #axes：2次元配列(要素数を指定してサブプロットを操作できる)
    #print(axes)
    """
    pyplot.subplotsのオプション
    nrows
        サブプロットの行数
    ncols
        サブプロットの列数
    shareⅹ
        全てのサブプロットで同じんのメモリを使うように指定
        (xlimの変更がすべてのサブプロットに影響)
    sharey
        ↾のyバージョン
    subplot_kw
        各サブプロットを作成するために呼び出されるadd_subplotに渡される
        キーワード引数のディクショナリ
    **fig_kw
        subplotに与える、作図の際に用いる追加キーワード引数
        (plt.subplots(2, 3, figsize=(8, 6))など)

    """

    #fig.subplots_adjust(left = None, bottom = None, right = None, top = None,
    #                        wspace = None, hspace = None)
                            #wspace, hspace:図の幅と高さのうちサブプロット間の
                            #スペースとして使う領域の割合の指定


    #ax.cla()


    plt.show() 
    return





def main():
    nine1()


    return

main()