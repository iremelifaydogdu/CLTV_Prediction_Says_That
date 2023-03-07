#Customer Lifetime Value Prediction (Müşteri Yaşam Boyu Değeri Tahmini)
#Zaman projeksiyonlu olasılıksal lifetime value tahmini


# satın alma sayısı * satın alma başına ortalama kazanç

# CLTV=(Customer Value / Churn Rate)*Profit Margin

# Customer Value = Purchase Frequency * Average Order Value

#CLTV = Expected Number of Transaction * Expected Average Profit Margin

#Başlarında conditional var aslında expected'ın.
#Conditional=koşullu=kişi özelinde biçimlendirecek şeklinde
#Buradaki değerler daha önce görmüş olduğumuz değerlerin olasılıksal halleri
#olasılıksal dağılım yolları kullanarak tahminler yapmayı sağlayacaktır.
#bütün kitlenin satın alma alışkanlıklarını bir olasılık dğaılım kullanarak modelliycez, daha sonra
#bu olasılık dağılım kullanarak modellediğimiz davranış biçimlerini koşullu yani conditional yani kişi özelinde biçimlendirecek
#şekilde kullanarak her bir kişi için beklenen satın alma ve beklenen işlem sayılarını tahmin edicez.


#CLTV = BG/NBD Model * Gamma Gamma Submodel

##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması

#Expected: Bir rassal değişkenin beklenen değeri, o rassal değişkenin ortalaması demek

#Rassal: Değerlerini bir deneyin sonuçlarından alan değişkene denir.

#Bir değişkenin ortalamasını aldığımızda bu değişkenin ortalaması elimizde olur oldukça basit, bir rassal değişken demek bir değişkenin belirli bir olasılık dağılım izlediğini varsaydığımızda aslında o olasılık dağılımı izlediğini varsaydığımız değişkenin ortalaması demektir.
#Genel olarak kitleden bir dağılım yapısı öğrenicez ki bu dağılım yapısı insanların satın alma,transaction davranışlarının yapısı olacak. Bu olasılık dağılımın bir beklenen değeri (ortalaması) vardır.
#bu olasılık dağılımının beklenen değerini koşullandırarak yani bireyler özelinde biçimlendirerek her bir birey için beklenen işlem sayısını tahmin etmiş olucaz.

#Expected Number of Transaction

#Beklenen satış sayısında tahmin etmek=Expexted Sales Forecasting= Expected Number of Transaction yerine denebilirdi.

#birisinin satın alma sayısını tahmin edersem başlı başına zaten önemli bir iş yapmış olurum. Buna göre pazarlama elemanlarını planlayabilirim. Mail listelerimi ayarlayabilirim çünkü yapacağımız model sayesinde
#önümüzdeki bir hafta içerisinde satın alacağımız kişileri belirleyeceğiz beraber
#1 hafta içinde satın alma ihtimali en yüksek olanlar, hatta alacağı miktarları da tahminleyeceğiz birlikte.
# 1 ay 3 ay gibi.

#BG/NBD :namı diğer: Buy Till You Die (Ölene dek satın al)
#satın alcaz bir de bunu bırakıcaz. bu model bunu modelliyor.


#BG/NBD modeli Expected number of transaction için iki süreci olasılıksal olarak modeller.

#Transaction process (İşlem süreci yani satın alma sürecini modeller.)
#Dropout Process (bırakma , düşme , inaktif olma =Till you die)

#Transaction süreci: alive olduğu sürece , belirli bir zaman periyodunda, bir müşteri tarafından geröçekleştirilecek işlem syısı transaction rate parametresi ile poisson sağılır.

#Bir müşteri alive olduğu sürece kendi transaction rate'i etrafnda rastgele satın alma yapmaya devam edecektir.

#Bir kullanıcı bu ortalama satın alma davranışını ortalamasında (kendi satın alma davranışı etrafında belirli bir olasılıkta) olduğu gibi sürdürmeye devam eder.

#Transaction rate (satın alma sayısı) poisson dağılmış.

#**Transaction rate'ler her bir müşteriye göre değişir ve tüm kitle için gamma sağılır (r, a)

#Satın alma süreçleri, işlem oranları bütün bu kitle açısından farklılıklar gösteriyor.
#Dağılımını bildiğiniz bir kitlenin üzerinden çıkarımlarda bulunabilirsiniz. Olasılıksal tahminlerde bulunabilirsiniz.
#Eğer ben bu kitlenin genel dağılımını biliyorsam bunu koşullayarak bir bireyin özelliklerine indirgediğimde o bireyin ne kadar satın alma yapabileceğini tahmin edebilirim.

#Özetle BG/NBD modelinin iki modelinin iki süreci var
#1. süreci BUY satın alma sürecini modelliyor.
#2. süreci =Dropout süreci (till you die), Her bir müşterinin p olasılığı ile dropout rate (dropout probability)'i vardır.

#**Dropout rate'ler her bir müşteriye göre değişir ve tüm kitle için beta dağılır. (a, b)

#x değeri: bir müşterinin tekrar eden satış sayısıdır, yani en az ikinci kez işlem yapma durumu

#tx değeri: Müşteri Recency değeri (haftalık) = recency değeridir. Bir müşterinin ilk satın almasıyla son satın alması arasında geeçn süredir. (haftalık cinsten)
#Dikkat!: tx today date'e göre değil müşteri özelinde bir recency'dir. RFM analizindeki recency ile aynı şeyler değildir.
#Her bir müşterinin ilk ve son satın alması arasındaki farktır

#T: Müşteri yaşı (haftalık) = müşterilerin ilk satın alması arasından geçen zamandır. Yani müşterinin yaşıdır.

#elimizdeki her bir müşterinin kendi içinde değişen değerleri: x, tx, T

#r ve a ifadesi :gamma dağılımının parametreleridir. müşteriler arası işlem oranının farklılığını modelleyen transaction rate'i modelleyen gamma dağılımının parametreleridir.
#diğer a ve b ifadeleri ise inaktif olasılığı, droprate olasılığını modelleyen betanın parametreleridir.

#bunlar bu iki olasılık dağılımının parametreleridir. bu parametreleri tahmin etmek için maksimum olabilirlik yöntemi kullanılmaktadır.
#bizim bunları tahmin etmek gibi bir kaygımız/derdimiz yoktur.
#kullanacak olduğumuz fonksiyonlar aracılığıyla bunları basitçe fonksiyonlar zaten tahmin etmiş olucak.
#bu parametreler bizim elimizdeki verideki tüm kullanıcılarımız göz önünde bulundurularak elde edilecek olan satın alma davranışının olasılıksal halini biçimlendirecek olan parametrelerdir.
#Öyle parametreler bulucaz ki veri setimizdeki müşterilerimizin satın alma karakteristiğinin olasılıksal dağılımı olan gamma dağılımını bizim kitlemize en uygun şekilde biçimlendirecek parametreler olacaktır.

#r, a, a, b = kitlemizden öğrenecek olduğumuz olasılık dağılımı parametreleri
#x, tx, T = her bir bireyin özelinde özelleştirecek olduğumuz değerler

#CLTV= Expected Number of Transaction * Expected Average Profit
#CLTV = BG/NBD Model * Gamma Gamma Submodel

#işlem başına ortalama kazanç : Gamma Gamma Submodel: average profit kısmını olasılıksal olarak modelleyecek olan model
#satın alma sıklığı: BG/NBD Model : olasılıksal olarak modelleyecek olan model

##########################
#Gamma Gamma Submodel
##############################

#Bir müşterinin işlem başına ortalama ne kadar kar getirebileceğini tahmin etmek için kullanılır.
#Average order value değerinin olasılıksal hali

#*Bir müşterinin işlemlerinin parasal değeri (monetary) transaction value'larının ortalaması etrafında rastgele dağılır.
#ortalama transaction value , zaman içinde kullanıcılar arasında değişebilir fakat tek bir kullanıcı için değişmez
#ortalama transaction value tüm müşteriler arasında gamma dağılır.

#x: frequency değeri: tekrar eden satış sayısı, 2. kez satış yapmış olma durumunu ifade ediyor
#mx: monetary: gözlemlenen transaction value'larıdır. ( total price/toplam işlem sayısı )
#diğer parametreler ise dağılımdan gelecek olan dağılımla ilgili parametrelerdir.






##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################

!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None) #bütün sütunları göster
pd.set_option('display.width', 500) #bütün sütunları göster ama aşağı inmeden yan yana 500 kadar göster
pd.set_option('display.float_format', lambda x: '%.4f' % x) #virgülden sonra 4. basamağa kadar göster
from sklearn.preprocessing import MinMaxScaler #lifetime value değeri hesaplandıktan sonra bunu 0-1 ya da 0-100 gibi değerler arasına da çekmek isterseniz MinMaxScaler methoduyla

#kuracak olduğumuz modeller olasılıksal-istatistiksel modeller olduğundan dolayı bu modelleri kurarken kullanacak olduğumuz değişkenlerin dağılımları sonuçları direkt etkileyebilecektir. Bundan dolayı
#elimizdeki değişkenleri oluşturduktan sonra bu değişkenlerdeki aykırı değerlere bir dokunmamız gerekmektedir. Bu sebeple butplot ya da iqr yöntemi olarak geçen bir yöntem aracılığıyla önce aykırı değerleri tespit edicez
#daha sonra aykırı değerlerle mücadele kapsamında değerlendirilebilecek aykırı değerleri baskılama yöntemiyle belirlemiş olduğumuz aykırı değerleri belirli bir eşik değeriyle değiştiricez.
#bu işlemi yapmak için 2 tane fonksiyona ihtiyacımız var:

#1. fonksiyonumuz: Kendisine girilen değişken için eşik değer belirlemektir. Bu ne anlama gelir?
#aykırı değer bir değişkenin genel dağılımının oldukça dışında olan değerlerdir.
#aykırı değerleri baskılamak için eşik değer belirlemek gerekir.

#quantile fonksiyonu çeyreklik hesaplamak için kullanılır: değişkeni küçükten büyüğe sıralarız, yüzdelik olarak %5. %25. %90.değerleri seçtiğimizde bunlar çeyreklik değerleri demektir.
#normalde boxplot yönteminde %25 ve %75 seçilirken burada neden %1 ve %99 seçildi?

#bu Vahit hocanın kişisel yorumu bu projeyi bildiği için bu şekilde geliştirmiş: bu şu anlama geliyor:
#elimizdeki veri setini tekilleştirdiğimizde ve kullanıcı özeline indirgediğimizde
#buradaki transaction/işlem sayıları ve price'lar çok çok yüksek sayılarda ve çok yüksek frekanslarda değil.
#klasik boxplot yöntemiyle belirlemiş olduğumuz eşik değere silme methodunu uygularsak çok fazla veri silinecek


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#########################
# Verinin Okunması
#########################

df_ = pd.read_excel("HAFTA 3/Materyaller/cltv_prediction/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
#                  count       mean       std         min        25%        50%        75%        max
#Quantity    541910.0000     9.5522  218.0810 -80995.0000     1.0000     3.0000    10.0000 80995.0000
#Price       541910.0000     4.6111   96.7598 -11062.0600     1.2500     2.0800     4.1300 38970.0000
#Customer ID 406830.0000 15287.6842 1713.6031  12346.0000 13953.0000 15152.0000 16791.0000 18287.0000

df.head()
#  Invoice StockCode                          Description  Quantity         InvoiceDate  Price  Customer ID         Country
#0  536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6 2010-12-01 08:26:00 2.5500   17850.0000  United Kingdom
#1  536365     71053                  WHITE METAL LANTERN         6 2010-12-01 08:26:00 3.3900   17850.0000  United Kingdom
#2  536365    84406B       CREAM CUPID HEARTS COAT HANGER         8 2010-12-01 08:26:00 2.7500   17850.0000  United Kingdom
#3  536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6 2010-12-01 08:26:00 3.3900   17850.0000  United Kingdom
#4  536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6 2010-12-01 08:26:00 3.3900   17850.0000  United Kingdom

df.isnull().sum()
#Invoice             0
#StockCode           0
#Description      1454 #az sayıda olduğu için göz ardı edicez, silicez
#Quantity            0
#InvoiceDate         0
#Price               0
#Customer ID    135080 #silicez direkt
#Country             0
#dtype: int64


#########################
# Veri Ön İşleme
#########################

df.dropna(inplace=True)
df.describe().T #bakalım bir şeyler değişmiş olacak mı?

df = df[~df["Invoice"].str.contains("C", na=False)] #veri setinde invoice'ta başında c ifade olanları sildik. Çünkü başında c olanlar iadeleri içermektedir.


df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################
#BG/NBD ve Gamma Gamma modellerinin bizden bekledikleri özel bir veri yapısı var ve bunu hazırlamamız bekleniyor.
#transaction datasını yani işlem verisini kendi dinamikleri çerçevesinde bazı ön işleme dinamiklerinden geçirdik. Belli bir hazırlığını tamamladık.
#şimdi ise bu verinin üzerinden kullanıcılara göre tekilleştirilmiş, kullanıcılar özelinde toplulaştırılmış
#ve lifetimes modülünün / kütüphanesinin fonksiyonlarına yanıt verecek bir forma dönüştürmemiz gerekmektedir.

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde dinamiktir, min ve max satın alma tarihleri üzerinden hesaplanır her kullanıcı için)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç (daha önce farklıydı, toplam kazançtı ama burada monetary değerini ortalamasını alarak kullanıcaz)



cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_df
#            InvoiceDate             Invoice TotalPrice
#             <lambda_0> <lambda_1> <lambda>   <lambda>
#Customer ID
#12346.0000            0        326        1   310.4400
#12347.0000          365        368        7  4310.0000
#12348.0000          282        359        4  1770.7800
#12349.0000            0         19        1  1491.7200
#12350.0000            0        311        1   331.4600
#                 ...        ...      ...        ...
#18280.0000            0        278        1   180.6000
#18281.0000            0        181        1    80.8200
#18282.0000          118        127        2   178.0500
#18283.0000          333        338       16  2094.8800
#18287.0000          158        202        3  1837.2800
#[4338 rows x 4 columns]

#Gelen sonucun okunabilirliği biraz düşük bunu daha iyi okuyabilmek adına


cltv_df.columns = cltv_df.columns.droplevel(0) #üstteki başlıkları silmek için, kendim istiyorum
cltv_df
#            <lambda_0>  <lambda_1>  <lambda>  <lambda>
#Customer ID
#12346.0000            0         326         1  310.4400
#12347.0000          365         368         7 4310.0000
#12348.0000          282         359         4 1770.7800
#12349.0000            0          19         1 1491.7200
#12350.0000            0         311         1  331.4600
#                 ...         ...       ...       ...
#18280.0000            0         278         1  180.6000
#18281.0000            0         181         1   80.8200
#18282.0000          118         127         2  178.0500
#18283.0000          333         338        16 2094.8800
#18287.0000          158         202         3 1837.2800
#[4338 rows x 4 columns]


cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df
#             recency    T  frequency  monetary
#Customer ID
#12346.0000         0  326          1  310.4400
#12347.0000       365  368          7 4310.0000
#12348.0000       282  359          4 1770.7800
#12349.0000         0   19          1 1491.7200
#12350.0000         0  311          1  331.4600
#              ...  ...        ...       ...
#18280.0000         0  278          1  180.6000
#18281.0000         0  181          1   80.8200
#18282.0000       118  127          2  178.0500
#18283.0000       333  338         16 2094.8800
#18287.0000       158  202          3 1837.2800
#[4338 rows x 4 columns]


cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] #işlem başına ortalama kazanç değerleri
cltv_df
#             recency    T  frequency  monetary
#Customer ID
#12346.0000         0  326          1  310.4400
#12347.0000       365  368          7  615.7143
#12348.0000       282  359          4  442.6950
#12349.0000         0   19          1 1491.7200
#12350.0000         0  311          1  331.4600
#              ...  ...        ...       ...
#18280.0000         0  278          1  180.6000
#18281.0000         0  181          1   80.8200
#18282.0000       118  127          2   89.0250
#18283.0000       333  338         16  130.9300
#18287.0000       158  202          3  612.4267
#[4338 rows x 4 columns]



cltv_df.describe().T
#              count     mean      std    min      25%      50%      75%       max
#recency   4338.0000 130.4486 132.0396 0.0000   0.0000  92.5000 251.7500  373.0000
#T         4338.0000 223.8310 117.8546 1.0000 113.0000 249.0000 327.0000  374.0000
#frequency 4338.0000   4.2720   7.6980 1.0000   1.0000   2.0000   5.0000  209.0000
#monetary  4338.0000 364.1185 367.2582 3.4500 176.8512 288.2255 422.0294 6207.6700



cltv_df = cltv_df[(cltv_df['frequency'] > 1)] #frequency'i 1'den büyük duruma getirmek
cltv_df.describe().T


cltv_df["recency"] = cltv_df["recency"] / 7 #recency değerini haftalık cinse çevirelim
cltv_df["recency"]
#Customer ID
#12347.0000   52.1429
#12348.0000   40.2857
#12352.0000   37.1429
#12356.0000   43.1429
#12358.0000   21.2857
#               ...
#18272.0000   34.8571
#18273.0000   36.4286
#18282.0000   16.8571
#18283.0000   47.5714
#18287.0000   22.5714
#Name: recency, Length: 2845, dtype: float64


cltv_df["T"] = cltv_df["T"] / 7 #müşteri yaşı değerini haftalık cinse çevirelim
cltv_df["T"]
#Customer ID
#12347.0000   52.5714
#12348.0000   51.2857
#12352.0000   42.4286
#12356.0000   46.5714
#12358.0000   21.5714
#               ...
#18272.0000   35.2857
#18273.0000   36.8571
#18282.0000   18.1429
#18283.0000   48.2857
#18287.0000   28.8571
#Name: T, Length: 2845, dtype: float64


##############################################################
# 2. BG-NBD Modelinin Kurulması
##############################################################

#Model kurma işlemi birkaç satırlık koddan ibaret, veri ön hazırlama daha zordur.

bgf = BetaGeoFitter(penalizer_coef=0.001) #bir model nesnesi oluşturucam bu model nesnesi aracılığıyla sen fit methodunu kullanarak
# bana frequency, recency ve müşteri yaşı değerlerini verdiğinde sana bu modeli kurmuş olcam.

#burada ben gamma ve beta dağılımlarını kullanıyorum. buradaki parametreleri bulurken en çok olabilirlik yönteminden yararlanıcam.
#parametre işlemleri sırasında bir argümana ihtiyacım var:

#argümanımız: penalizer_coef=0.001 : Bu modelin parametrelerinin bulunması aşamasında katsayılara uygulanacak olan ceza katsayısıdır.
# makine öğrenmesi kapsamında ele alınacak olan konularda parametre tahmin yöntemlerinden bazıları daha detaylı bir şekilde ele alındığında oradaki detayları girmek faydalı olabilir.
#şuanda konumuz kapsamında bg/nbd modeli bize en çok olabilirlik yöntemi ile beta ve gamma dağılımlarının parametrelerini bulmakta ve bir tahmin yapabilmemiz için ilgili modeli oluşturmaktadır.


bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])
#<lifetimes.BetaGeoFitter: fitted with 2845 subjects, a: 0.12, alpha: 11.41, b: 2.49, r: 2.18>


################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################
#model nesnem bf isi , fit ettik, bi fonksiyonumuz var: conditional_expected_number_of_purchases_up_to_time, teorisinde transaction yazıyordu.
#ve belirli bir zaman periyodu olduğu ifade edilmiş.=1 (haftalık cinsten oluşturduğumuzdan dolayı)
#1 haftalık tahmin yap, bütün müşterilerimiz için, tüm tahminlerin sonucunu azalan şekilde gönder.



bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)
#Customer ID
#12748.0000   3.2495
#14911.0000   3.1264
#17841.0000   1.9402
#13089.0000   1.5374
#14606.0000   1.4639
#15311.0000   1.4336
#12971.0000   1.3569
#14646.0000   1.2064
#13408.0000   0.9862
#18102.0000   0.9685
#dtype: float64

#yukarıdaki işlemin aynısını predict fonksiyonu ile de yapabiliriz.
#scikit-learn ara yüzü ile predict: (tahmin etme işlemleri)
#bg/nbd modeli için bu method(predict) geçerlidir ama gamma gamma için geçerli değildir.

#bütün müşteriler için 1 hafta içerisinde beklediğimiz satın almaları hesaplayıp bunu cltv dataframe veri setimize ekleyelim.


bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)
#Customer ID
#12748.0000   3.2495
#14911.0000   3.1264
#17841.0000   1.9402
#13089.0000   1.5374
#14606.0000   1.4639
#15311.0000   1.4336
#12971.0000   1.3569
#14646.0000   1.2064
#13408.0000   0.9862
#18102.0000   0.9685
#dtype: float64



cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])
cltv_df["expected_purc_1_week"]
#Customer ID
#12347.0000   0.1413
#12348.0000   0.0920
#12352.0000   0.1824
#12356.0000   0.0862
#12358.0000   0.1223
#              ...
#18272.0000   0.1721
#18273.0000   0.1043
#18282.0000   0.1357
#18283.0000   0.3017
#18287.0000   0.1208
#Name: expected_purc_1_week, Length: 2845, dtype: float64



###############################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

#1 ay 4 haftadır:

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)
#Customer ID
#12748.0000   12.9633
#14911.0000   12.4722
#17841.0000    7.7398
#13089.0000    6.1330
#14606.0000    5.8399
#15311.0000    5.7191
#12971.0000    5.4131
#14646.0000    4.8119
#13408.0000    3.9341
#18102.0000    3.8636
#dtype: float64



cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

cltv_df["expected_purc_1_month"] #1 ayda beklenen satışları veri setime ekledim
#Customer ID
#12347.0000   0.5635
#12348.0000   0.3668
#12352.0000   0.7271
#12356.0000   0.3435
#12358.0000   0.4862
#              ...
#18272.0000   0.6856
#18273.0000   0.4157
#18282.0000   0.5392
#18283.0000   1.2034
#18287.0000   0.4810
#Name: expected_purc_1_month, Length: 2845, dtype: float64

#şirketim adına 1 ay içerisinde ne kadar beklenen satış sayısı var?:

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()
#1776.893473220295


################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()
#5271.112433826363

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################


#yapmış olduğumuz tahminlerin başarısını nasıl değerlendiricez?


plot_period_transactions(bgf)
plt.show()

#gerçek değerler maviler, tahmin edilenler turuncu.


##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

#bg/nbd satın alma sayısını modelliyordu, gamma gamma ise average profit'i modelliyordu



ggf = GammaGammaFitter(penalizer_coef=0.01) #model nesnemi çağırıyorum , ggf diye bir isimlendirme ile
#bu model nesnesini kullanarak model kur bilgisini vermem lazım

ggf.fit(cltv_df['frequency'], cltv_df['monetary']) #ggf nesnesini fit et, model kur diyorum o da bana frequency ve monetary değerlerini ver diyor.
#model üstteki 2 satır kod ile kurulmuş oldu. Bu işin teorisini ve veri ön işleme kısmını anlamak önemli.

#<lifetimes.GammaGammaFitter: fitted with 2845 subjects, p: 3.79, q: 0.34, v: 3.73> :parametrelerle ilgili bulduğu değerleri çıktıda verdi.

#GammaGammaFitter nesnesi bu parametreleri buluyor.
#bizim ana amacımız purchase frequency modelledik average profiti modellemeye çalışıyorduk

#dolayısıyla bu modelimizin bize sunacağı şey koşullu beklenen average profit değerleri olacak

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)
#Customer ID
#12347.0000    631.9123
#12348.0000    463.7460
#12352.0000    224.8868
#12356.0000    995.9989
#12358.0000    631.9022
#12359.0000   1435.0385
#12360.0000    933.7905
#12362.0000    532.2318
#12363.0000    304.2643
#12364.0000    344.1370
#dtype: float64

#azalan bir şekilde gözlemlemek istersek:

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

#şuanda bütün müşterilerim içerisinde bana beklenen karı, expected average'ı, ortalama karı getirmiş oldu her bir müşteri için


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)
#             recency       T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit
#Customer ID
#12415.0000   44.7143 48.2857         21 5724.3026                0.3796                 1.5139                 4.5080                5772.1782
#12590.0000    0.0000 30.2857          2 4591.1725                0.0115                 0.0460                 0.1363                5029.4196
#12435.0000   26.8571 38.2857          2 3914.9450                0.0763                 0.3041                 0.9035                4288.9440
#12409.0000   14.7143 26.1429          3 3690.8900                0.1174                 0.4674                 1.3854                3918.8128
#14088.0000   44.5714 46.1429         13 3864.5546                0.2603                 1.0379                 3.0896                3917.1297
#18102.0000   52.2857 52.5714         60 3859.7391                0.9685                 3.8636                11.5112                3870.9969
#12753.0000   48.4286 51.8571          6 3571.5650                0.1261                 0.5028                 1.4973                3678.5783
#14646.0000   50.4286 50.7143         73 3646.0757                1.2064                 4.8119                14.3340                3654.8148


#ufak bir değerlendirme yaparsak:
#*frekansı yüksek olduğu hade aşağıda olanlar var.
#*yaşı daha büyük olduğu halde aşağıda olanlar var.
#POTANSİYEL MÜŞTERİ DEĞERLERİNİ YAKALAMA İMKANI DA VAR.
#ELİMİZDE EXPECTED TRANSACTİON DEĞERLERİ VAR. 1 HAFTALIK 1 AYLIK VS.
#VE ORTALAMAM BEKLENEN KARLILIK DEĞERLERİ HER BİR MÜŞTERİ ÖZELİNDE HESAPLANDI.


##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################
#BG/NBD ile beklenen frekansları beklenen işlemleri modelledik
#Gamma gamma modeli ile beklenen karlılıkları modelledik.

#Bu ikisini bir araya getirdiğimizde temel formülümüzdeki çarpma işlemini gerçekleştiriyor ve CLTV değerlerini hesaplayabiliyor olucaz.

#model nesnemizi getirdik daha önce kurduğumuz gamma gamma modeli: ggf, daha önce kurduğumuz bg/nbd modeli: bgf


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi. bunu recency ve müşteri yaşı açısından bekliyorum.
                                   discount_rate=0.01)

cltv.head()
#Customer ID
#12347.0000   1128.4477
#12348.0000    538.8089
#12352.0000    517.5000
#12356.0000   1083.0903
#12358.0000    966.6727
#Name: clv, dtype: float64



cltv = cltv.reset_index() #satırdaki customer id'yi sütuna geçirmek için, indexlerde olan customer_id şuanda değişeken dönüştü.

cltv_df #bunun içerisinde kademe kademe bütün işlemlerimiz yer alıyor

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left") #left join ile yani cltv_df'e göre birleştirme işlemini gerçekleştir ve head atmak yerine:

cltv_final.sort_values(by="clv", ascending=False).head(10) #DİKKAT!: hep cltv dedik burada clv değeri var. customer_lifetime_value fonksiyonu return ettiği  dataframe'deki isimlendirmeyi böyle yapmış.
#      Customer ID  recency       T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit        clv
#1122   14646.0000  50.4286 50.7143         73 3646.0757                1.2064                 4.8119                14.3340                3654.8148 55741.0845
#2761   18102.0000  52.2857 52.5714         60 3859.7391                0.9685                 3.8636                11.5112                3870.9969 47412.5801
#843    14096.0000  13.8571 14.5714         17 3163.5882                0.7287                 2.8955                 8.5526                3196.4361 29061.6614
#36     12415.0000  44.7143 48.2857         21 5724.3026                0.3796                 1.5139                 4.5080                5772.1782 27685.1000
#1257   14911.0000  53.1429 53.4286        201  691.7101                3.1264                12.4722                37.1641                 692.3264 27377.4115
#2458   17450.0000  51.2857 52.5714         46 2863.2749                0.7474                 2.9815                 8.8830                2874.1987 27166.0643
#874    14156.0000  51.5714 53.1429         55 2104.0267                0.8775                 3.5005                10.4298                2110.7542 23424.4032
#2487   17511.0000  52.8571 53.4286         31 2933.9431                0.5088                 2.0298                 6.0476                2950.5801 18986.6123
#2075   16684.0000  50.4286 51.2857         28 2209.9691                0.4781                 1.9068                 5.6801                2223.8850 13440.4131
#650    13694.0000  52.7143 53.4286         50 1275.7005                0.8008                 3.1946                 9.5186                1280.2183 12966.1347

#3 aylık customer lifetime value değerleri geldi. bu değerlere göre şuanda 3 aylık bir plan yapabilirim.

#DİKKAT! burada bazı yorumlamalar yaparak daha yaından buradaki sonuçları anlamaya çalışıcaz, şuanda odaklanacağımız olduğumuz değerler cltv değeri, expected_average_profit , monetary, frequency, T(müşteri yaşı) ve recency değerleridir.
#burada daha sağlıklı bir değerlendirme yapmak istersek bu veri setine 3 aylık beklenen satışları da ekleyebiliriz.




##############################################################
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################

cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)
#      Customer ID  recency       T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit        clv segment
#1122   14646.0000  50.4286 50.7143         73 3646.0757                1.2064                 4.8119                14.3340                3654.8148 55741.0845       A
#2761   18102.0000  52.2857 52.5714         60 3859.7391                0.9685                 3.8636                11.5112                3870.9969 47412.5801       A
#843    14096.0000  13.8571 14.5714         17 3163.5882                0.7287                 2.8955                 8.5526                3196.4361 29061.6614       A
#36     12415.0000  44.7143 48.2857         21 5724.3026                0.3796                 1.5139                 4.5080                5772.1782 27685.1000       A
#1257   14911.0000  53.1429 53.4286        201  691.7101                3.1264                12.4722                37.1641                 692.3264 27377.4115       A
#2458   17450.0000  51.2857 52.5714         46 2863.2749                0.7474                 2.9815                 8.8830                2874.1987 27166.0643       A
#874    14156.0000  51.5714 53.1429         55 2104.0267                0.8775                 3.5005                10.4298                2110.7542 23424.4032       A
#2487   17511.0000  52.8571 53.4286         31 2933.9431                0.5088                 2.0298                 6.0476                2950.5801 18986.6123       A
#2075   16684.0000  50.4286 51.2857         28 2209.9691                0.4781                 1.9068                 5.6801                2223.8850 13440.4131       A
#650    13694.0000  52.7143 53.4286         50 1275.7005                0.8008                 3.1946                 9.5186                1280.2183 12966.1347       A
#841    14088.0000  44.5714 46.1429         13 3864.5546                0.2603                 1.0379                 3.0896                3917.1297 12875.7762       A
#1754   16000.0000   0.0000  0.4286          3 2335.1200                0.4220                 1.6639                 4.8439                2479.8048 12751.9305       A
#1441   15311.0000  53.2857 53.4286         91  667.7791                1.4336                 5.7191                17.0411                 669.0960 12132.2867       A
#373    13089.0000  52.2857 52.8571         97  606.3625                1.5374                 6.1330                18.2736                 607.4877 11811.7794       A
#1324   15061.0000  52.5714 53.2857         48 1120.6019                0.7717                 3.0786                 9.1730                1124.7458 10977.9341       A
#949    14298.0000  50.2857 51.5714         44 1162.8627                0.7277                 2.9026                 8.6470                1167.5522 10742.1077       A
#1647   15769.0000  51.8571 53.1429         26 1873.6702                0.4329                 1.7268                 5.1449                1886.4041 10326.6579       A
#2774   18139.0000   0.0000  2.7143          6 1406.3900                0.5289                 2.0904                 6.1111                1448.9171  9403.5742       A
#2652   17841.0000  53.0000 53.4286        124  330.1344                1.9402                 7.7398                23.0625                 330.6271  8113.3777       A
#2699   17949.0000  52.8571 53.1429         45  848.4303                0.7280                 2.9040                 8.6524                 851.7980  7842.0081       A
#698    13798.0000  52.8571 53.2857         57  650.9085                0.9112                 3.6351                10.8310                 652.9618  7525.1113       A
#2422   17389.0000  47.1429 47.4286         34  933.4722                0.6119                 2.4403                 7.2663                 938.3712  7254.5601       A
#1772   16029.0000  47.8571 53.4286         63  922.1319                0.6115                 2.4393                 7.2681                 924.7383  7151.4977       A
#379    13098.0000  45.1429 45.4286         28 1031.5157                0.5280                 2.1055                 6.2677                1038.0825  6922.3111       A
#298    12931.0000  48.0000 51.1429         15 1993.1667                0.2697                 1.0758                 3.2043                2016.7466  6875.8501       A
#1763   16013.0000  52.4286 53.1429         47  697.1070                0.7580                 3.0238                 9.0095                 699.7700  6708.2553       A
#1960   16422.0000  50.1429 52.8571         51  651.6810                0.8086                 3.2258                 9.6111                 653.9793  6687.8900       A
#2430   17404.0000  50.8571 51.5714         15 1931.6103                0.2702                 1.0777                 3.2102                1954.4698  6675.6882       A
#369    13081.0000  51.2857 53.1429         11 2576.1255                0.2012                 0.8026                 2.3909                2617.7613  6659.5248       A
#1683   15838.0000  50.8571 52.5714         19 1541.4600                0.3274                 1.3060                 3.8905                1555.8664  6440.6501       A
#215    12748.0000  53.1429 53.4286        209  154.9302                3.2495                12.9633                38.6276                 155.0768  6373.8608       A
#1141   14680.0000  49.4286 53.1429         16 1796.9641                0.2759                 1.1006                 3.2789                1816.9028  6338.7935       A
#217    12753.0000  48.4286 51.8571          6 3571.5650                0.1261                 0.5028                 1.4973                3678.5783  5860.1541       A
#31     12409.0000  14.7143 26.1429          3 3690.8900                0.1174                 0.4674                 1.3854                3918.8128  5773.5565       A
#511    13408.0000  53.0000 53.4286         62  453.5006                0.9862                 3.9341                11.7223                 454.8338  5673.0939       A
#691    13777.0000  53.1429 53.4286         33  757.7258                0.5400                 2.1540                 6.4178                 761.8450  5202.4648       A
#2659   17857.0000  51.5714 52.2857         23  886.8600                0.3925                 1.5655                 4.6638                 893.7655  4435.1214       A
#209    12731.0000  48.7143 52.1429         12 1574.6592                0.2187                 0.8724                 2.5987                1598.0798  4418.7633       A
#1102   14607.0000  35.0000 37.4286         14 1055.8329                0.3253                 1.2963                 3.8540                1069.3540  4384.1550       A
#2280   17084.0000   0.0000  5.1429          2 1474.8750                0.2168                 0.8578                 2.5124                1617.0440  4315.3587       A
#2570   17675.0000  52.2857 52.5714         31  657.2348                0.5160                 2.0582                 6.1319                 661.0556  4313.0270       A
#740    13881.0000  44.0000 44.5714         20  857.9760                0.3931                 1.5675                 4.6654                 865.6736  4296.7863       A
#72     12477.0000  39.7143 43.1429          6 2203.2900                0.1459                 0.5816                 1.7303                2269.5515  4177.6237       A
#67     12471.0000  51.7143 52.1429         30  647.2723                0.5036                 2.0087                 5.9841                 651.1632  4146.0660       A
#49     12435.0000  26.8571 38.2857          2 3914.9450                0.0763                 0.3041                 0.9035                4288.9440  4122.1484       A
#102    12536.0000   2.1429  8.5714          3 1451.2767                0.2159                 0.8559                 2.5160                1541.6976  4121.5330       A
#1873   16240.0000   7.5714 11.1429          2 1751.4600                0.1728                 0.6858                 2.0189                1919.9072  4119.1447       A
#1350   15113.0000   6.2857  7.8571          3 1203.1100                0.2575                 1.0206                 2.9984                1278.2947  4072.2726       A
#1311   15039.0000  51.2857 52.7143         47  420.4677                0.7607                 3.0343                 9.0405                 422.1055  4060.3618       A
#1374   15159.0000  51.2857 51.7143         30  609.6417


cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})
#        Customer ID                     recency                        T                  frequency             monetary                   expected_purc_1_week                expected_purc_1_month                expected_purc_3_month                 expected_average_profit                         clv
#               mean count           sum    mean count        sum    mean count        sum      mean count   sum     mean count         sum                 mean count      sum                  mean count      sum                  mean count       sum                    mean count         sum      mean count          sum
#segment
#D        15558.4761   712 11077635.0000 22.0738   712 15716.5714 40.4649   712 28811.0000    3.0646   712  2182 183.9631   712 130981.7123               0.0711   712  50.5898                0.2830   712 201.4774                0.8400   712  598.0905                199.4377   712 141999.6650  143.2970   712  102027.4901
#C        15309.6343   711 10885150.0000 30.6697   711 21806.1429 38.1111   711 27097.0000    4.0956   711  2912 271.6948   711 193174.9847               0.1206   711  85.7309                0.4802   711 341.3936                1.4251   711 1013.2176                289.9856   711 206179.7461  380.7919   711  270743.0065
#B        15352.8186   711 10915854.0000 29.5148   711 20985.0000 34.8117   711 24751.1429    5.4416   711  3869 373.4425   711 265517.6312               0.1625   711 115.5066                0.6465   711 459.6926                1.9165   711 1362.6595                393.8944   711 280058.9431  688.2650   711  489356.3925
#A        14947.3586   711 10627572.0000 31.4109   711 22333.1429 34.4840   711 24518.1429   11.3586   711  8076 659.8586   711 469159.4581               0.2736   711 194.4993                1.0891   711 774.3298                3.2309   711 2297.1448                685.8998   711 487674.7239 2222.3600   711 1580097.9284



##############################################################
# 6. Çalışmanın Fonksiyonlaştırılması
##############################################################

def create_cltv_p(dataframe, month=3):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")
















































































































































