import numpy as np

def four1():
    data = np.random.randn(2,3) #2要素の配列＊２

    print(data)

    data *= 10  #各要素10倍

    print(data)


    data += data        #対応する要素同士の可算

    print(data)

    """
    numpyの特徴
        全ての要素が同じ型である必要がある
        shapeとdtyoeという属性を持ち、それぞれの配列変数(要素？)ごとに固有の値を持つ
            shape:その配列の次元ごとの要素数を格納するタプル:()でくくられるやつ
            dtype:配列要素に期待する型

    """

    print(data.shape)           #外側から要素数を教えてくれる

    print(data.dtype)           #型
    return

def fourOne1():

    data1 = [6, 7.5,8,0,1]

    arr1 = np.array(data1)                   #リストからndarrayを生成
                                            #リストのほかに、ndarray自身を含む
                                            #シーケンス型を引数にとれる

    print(arr1)

    data2 = [[1,2,3,4], [5,6,7,8]]

    arr2 = np.array(data2)                      #2次元配列
    print(data2)

    print(arr2.ndim)            #次元数

    print(arr2.shape)           #要素数のタプル


    print(arr1.dtype)           #要素の型を明示していない場合は最適な
                                #型を推測してくれる→dtypeに格納される
    print(arr2.dtype)

    print(np.zeros(10))     #要素数10の配列(中身すべて)を生成
    print(np.zeros((3,6)))  #3行6列で生成(引数はタプル)
                            #→shapeを引数にとってるともいえる
    
    print(np.empty((2,3,2)))    #empty:要素の各値は不定
    
    print(np.ones((3,3,3)))     #全要素1で生成

    print(np.arange(15))    #pythonのrangeと同じで0~14の値(15要素)の配列を生成
    
    """
    標準的なndarrayの生成関数
    array
        入力に、リスト、タプル、python配列、その他列挙型などのデータを受けて生成
        要素の型(dtype)は推測または明示されたもの
    asarray
        arrayと同様
        ただし入力がndarrayだったとき新規に変数を生成しない
        →(アドレス変えない的な？shallowCopy?)
    arange
        pythonのrangeと同じ要領で生成
    ones, ones_like
        ones:指定されたサイズのndarrayを生成(中身全部1)
        ones_like:引数に別のシーケンス型を受け，それをテンプレートとして要素を全て1に
                　→与えた配列の中身をすべて1で書き換える？
    zeros, zeros_like
        ↾の中身0バージョン
    empty, empty_like
        〃
    full, full_like
        full:指定されたサイズのndarrayを指定されたdtypeで生成→要素をすべて指定された値にする
        full_like:↾と同じ要領
    eye, identity
        identity:N*Nの単位行列を生成
        eye:↾を生成したうえで列数を指定して切り出せる
    """

    arr = np.zeros((2,2,2))
    print(arr)
    arrLike = np.ones_like(arr)         #arrを１で書き換えてるわけではない
                                        #金型だけ借りてるっぽい((2，2，2)の部分)
    print("arr", arr)
    print(arrLike)
    arr += 3
    print(arr)


    arrFull = np.full(2, 255)   #要素数2(値255)で初期化

    print(arrFull)

    arrFull2 = np.full((2,2,5), 255) #タプルを使って初期化

    print(arrFull2)



    return

def fourOne2():    #データ型と一括処理(行列演算)
    #dtype:メモリ上のデータ表現形式を示す特別なオブジェクト
    #データについてのデータ→メタデータ

    arr1 = np.array([1,2,3], dtype = np.float64)

    arr2 = np.array([1,2,3], dtype = np.int32)

    print(arr1.dtype)
    print(arr2.dtype)

    """
    numPyのデータ型

    int8, uint8
        整数3ビット、符号なし整数8ビット
    int16, uint16
    int32, uint32
    int64, uint64
    float16
    float32
        Cのfloatと同等
    float64
    float128
    complex64, complex128, complex256
        複素数。2つの浮動小数点数の組で表される
        64は2つの値は32ビット、128は64ビット、256は128ビット
    bool
        真偽値
    object
        pythonオブジェクト型
    strig_
        ※_は誤植ではない
        固定長文字列。_のところに数字が入る
        10文字ならstring10(S10)
    unicode_
        固定長ユニコード型。1文字当たりのバイト数はプラットフォームに依存
    """



    arr = np.array([1,2,3,4,5])

    print(arr.dtype)

    float_arr = arr.astype(np.float64)  #明示的な型変換

    print(float_arr.dtype)

    print("aaaaaaaaaaaaaaaaaa")

    arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10,1])

    print(arr)

    print(arr.astype(np.int8))      #浮動小数点数→整数で小数点以下切り捨て
                                    #-1.2とかは-1になる(-2にはならん)(0への丸め)

    print(arr.astype(np.uint8))     #マイナスはオ-バーフローする
                                    #-1.2　→　-1　→　255　の流れ(例外は投げてくれない)
    
    #array = np.array([1,2,3,4,5])
    #array.append(6)                #リストと同じ関数は使えないっぽい
    #print(array)

    numeric_strings = np.array(["1","2","3"], dtype = np.string_)

    print(numeric_strings.astype(float))#floatって書いてもいいのかよ
                                        #→標準python型を等価なdtypeに変換してくれる
    """
    numpy_stringへの変換は、文字データが固定長なので警告なしに切り捨てが起こる
    pandasを使うといいらしい
    """

    print("aaaaaaaaaaaaaaaaaaaaaa")

    int_array = np.arange(10)
    calibers = np.array([.22,.270,.357,.380,.44,.50], dtype = np.float64)

    print(int_array.astype(calibers.dtype))    #calibersの型に合わせる

    empty_uint32 = np.empty(8,dtype = "u4") #型コード指定
    print(empty_uint32)                     #int指定ならemptyでも整数になる(値は不定)
    
    """ 
    astypeを呼び出す際、必ず新規のndarrayが生成される
    """



    return

def fourOne3(): #算術演算
    
    arr = np.array([[1.,2.,3.],[4.,5.,6.,]])  #floatを明示してる？

    print(arr)

    array =  arr*arr      #対応する要素同士の掛け算
                            #(行列演算子の×ではない(この場合行列定義できないけど))


    print(array)

    array = arr-arr
    print(array)            #対応する要素同士の引き算(行列演算)

    array = 1 / arr         #スカラーとの算術演算
                            #要素ごとに加減乗除が計算される
    
    print(array)

    array = arr ** 0.5      #**は累乗 
    print(array)

    arr2 = np.array([[0.,4.,1.],[7.,2.,12.]])

    array = arr > arr2  #各要素に真偽値を格納
                        #中身が完全に変わる分にはデータ型変えてもいいっぽい
    print(array)


    return

def fourOne4():             #インデックス参照とスライシングの基礎
    arr = np.arange(10)

    print(arr)
    print(arr[5])   #要素の呼び出し

    print(arr[5:8]) #5~8番目の要素の切り出し(スライシング)

    arr[5:8] = 12   #5~8番目の値を12にする
    print(arr)

    #############中断################


    return

def main():

    #four1()
    #fourOne1()
    #fourOne2()
    #fourOne3()
    fourOne4()


    return

main()




















