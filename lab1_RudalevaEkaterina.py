#!/usr/bin/env python
# coding: utf-8

# # <center> Практические задания по цифровой обработке сигналов </center>
# # <center> Первая лабораторная работа </center>
# 

# В данной работе Вы познакомитесь с основными методами работы с аудиоданными в Python. Разбересь в том, как работает свертка, и примените пару интересных фильтров.

# # Задание 1. Работа с аудиофайлами в Python (1 балл)

# ## Теория
# 
# Звук - это аналоговый сигнал. То есть он является непрерывным по времени и по значениям. Для того, чтобы работать со звуком на цифровом устройстве, надо преобразовать его в цифровое представление. Для этого надо разделить непрерывный сигнал на промежутки времени (дискретизация сигнала) и разбить непрерывные значения на интервалы (квантование сигнала). Выбраные параметры дискретизации и квантования сигнала напрямую влияют на качество цифрового сигнала. 
# 
# 
# ## Практика
# 
# 1. Что хранится в .wav файле? Как узнать параметры дискретизации и квантования .wav файла? 
# 
# 2. Запишите аудиофайл со своим голосом. Загрузите его. Попробуйте поменять ему частоту дискретизации. Нарисуйте форму волны считанного файла. Воспроизведите полученные сигналы. При какой частоте дискретизации становится невозможно разобрать человеческую речь?   
# 
# 3. Чем .wav отличается от других кодеков, например .mp3 или .ogg?
# 
# 
# ### Подсказка
# 
# Записать цифровой сигнал можно при помощи, например, [Audacity](https://www.audacityteam.org) или [Adobe Audition](https://www.adobe.com/ru/products/audition.html). Для считывания файлов воспользуйтесь библиотекой [scipy](https://www.scipy.org) или [librosa](https://librosa.org/doc/latest/index.html). Для воспроизведения аудиофайла удобно использовать класс Audio из модуля IPython.display, а для отрисовки - matplotlib.

# ## Ответы

# ### 1
# В .wav файле хранится несжатый аудиопоток. Такой файл состоит из части заголовка файла , тела с аудиопотоком и хвоста для дополнительной информации, куда возможна запись собственных метаданных при помощи аудиоредакторов.
# В заголовочной части .wav файла  содержится информация о размере файла, количестве каналов, частоте дискретизации, количестве бит в сэмпле (квантовании).

# ### 2

# In[20]:


import matplotlib.pyplot as plt 
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.io import wavfile
import IPython.display as ipd
import scipy.signal as sps
import scipy 
import sys


# Запишите аудиофайл со своим голосом:
# https://github.com/EkaterinaRudaleva/lab1_dsp/blob/main/df_for_lab1.wav

# In[24]:


# Uploading the data
data_path='/df_for_lab1.wav'#YOUR PATH
sr, x = wavfile.read(data_path)
print("Sample rate:", sr, "Auodiofile-Array:" ,x)

#Duration
print( "Duration of audiofile: ", round(len(x)) / sr)

#play
ipd.Audio(data_path, autoplay=True)


# In[42]:


#Time array
t = np.linspace(0,x.shape[0]/sr,x.shape[0])

#Plot
fig, ax=plt.subplots()
ax.plot(t, x, linewidth = 0.05)
ax.set(xlabel= "Time in seconds", ylabel="Amplitude")
plt.title("Signal by default "+  str(sr)+" Hz")
plt.show()


# In[26]:


#Понизим частоту до 5 кГЦ

sr_new=5000 #Hz
# Resample the data
samle_num = round(len(x) * float(sr_new) / sr)
data = sps.resample(x, samle_num)
#Downloading new data
wavfile.write("./out.wav", sr_new,data.astype(np.int16))
ipd.Audio("./out.wav", autoplay=True)


# In[46]:



sr, x = wavfile.read("./out.wav")
print("New Sample rate:", sr)
t = np.linspace(0,x.shape[0]/sr,x.shape[0])

#Plotting
fig, ax=plt.subplots()
ax.plot(t, x, linewidth = 0.05)
ax.set(xlabel= "Time in seconds", ylabel="Amplitude")
plt.title("Signal "+ str(sr)+" Hz")
plt.show()


# In[ ]:


#Понизим частоту до 1кГЦ
sr_new1000=1000 #Hz
# Resample data
sample_num1000 = round(len(x) * float(sr_new1000) / sr)
data1000 = sps.resample(x, sample_num1000 )

wavfile.write("./out1000.wav", sr_new1000,data1000.astype(np.int16))
ipd.Audio("./out1000.wav", autoplay=True)
#Человеческая речь едва различима


# In[52]:


sr, x = wavfile.read("./out1000.wav")
print("Sample rate:", sr)
t = np.linspace(0,x.shape[0]/sr,x.shape[0])
t
fig, ax=plt.subplots()
ax.plot(t, x, linewidth = 0.05)
ax.set(xlabel= "Time in seconds", ylabel="Amplitude")
plt.title("Signal "+ str(sr)+" Hz")
plt.show()


# In[50]:



#Понизим частоту до 900Гц
sr_new1=900 #Hz
# Resample data
sample_num1 = round(len(x) * float(sr_new1) / sr)
data1 = sps.resample(x, sample_num1 )

wavfile.write("./out1.wav", sr_new1,data1.astype(np.int16))

ipd.Audio("./out1.wav", autoplay=True)
#Человеческая речь не различима


# In[51]:



sr, x = wavfile.read("./out1.wav")
print("Sample rate:", sr)
t = np.linspace(0,x.shape[0]/sr,x.shape[0])
t
fig, ax=plt.subplots()
ax.plot(t, x, linewidth = 0.05)
ax.set(xlabel= "Time in seconds", ylabel="Amplitude")
plt.title("Signal "+ str(sr)+" Hz")
plt.show()


# ### 3 
# В отличие от  .mp3 или .ogg .wav сохраняет качество звука, поскольку сохраняет его несжатым.

# # Задание 2. Гармонические сигналы (1 балл)

# ## Теория
# [Гармонические колебания](https://ru.wikipedia.org/wiki/Гармонические_колебания) -  колебания, при которых физическая величина изменяется с течением времени по гармоническому (синусоидальному/косинусоидальному) закону. 
# 
# В общем случае гармонические колебания задаются формулой:
# 
# $$y=A \cos(\omega t+\varphi_0)$$
# 
# где $А$ - это амплитуда, $\omega$ – циклическая частота (радиан/с), $\varphi$ - фаза (сдвиг), $t$ – время. 
# 

# In[21]:


# Сначала определим функцию для отрисовки сигнала с хорошим масштабом и сеткой
# Это поможет легче анализировать сигнал
def draw_signal(data, figsize=(14, 14)):
    plt.figure(figsize=figsize)
    plt.plot(data, linewidth=2)
    plt.minorticks_on()
    plt.xticks(np.arange(0, 1000, step=100))
    plt.yticks(np.arange((data.min().round())//10*10, 
                         (data.max().round())//10*10+10, step=5))
    plt.grid(which='major',
        color = 'k', 
        linewidth = 1)
    plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
    plt.show()


# In[22]:


# Читаем данные с подготовленными сигналами
import pickle
with open("./data.pickle", "rb") as f:
    test_data = pickle.load(f)
# Теперь можно приступать к практике!


# ## Практика
# 
# Постройте графики трех сигналов a, b и c из test_data['task2']. Попробуйте подобрать коэффициенты для этих сигналов. Сгенерируйте сигналы (1000 отсчетов) с подобранными коэффициентами. Постройте графики сгенерированных сигналов и пройдите тест на схожесть с оригинальным.
# 
# 
# Подсказка. Фаза, период и амплитуда сигнала - целочисленные. Для генерации пользуйтесь библиотекой numpy и функциями arange, sin, cos.

# In[23]:


draw_signal(test_data['task2']['a'])


# In[24]:



ts = np.linspace(0,1000,1000)

phi=0
import math as m


def form(A1=45,T=425, N=1000, del_t=105
        ):
    x= []
    for t in ts:
        x.append(A1 * np.sin(m.pi*2/T * t + (del_t)/T*2*m.pi))
    x=np.array(x)
    return x

a = form()
draw_signal(a)


# In[25]:


assert len(a) == 1000
assert np.allclose(a, test_data["task2"]["a"], atol=1)
print("Ok!")


# In[26]:


draw_signal(test_data['task2']['b'])


# In[34]:


# YOUR CODE HERE
ts = np.linspace(0,1000,1000)

phi=0
import math as m


def form(A1=50,T=100.1
         , N=1000, del_t=100.1
         
        ):
    x= []
    for t in ts:
        x.append(A1 * np.sin(m.pi*2/T * t + (del_t)/T*2*m.pi))
    x=np.array(x)
    return x

b= form()
draw_signal(b)


# In[35]:


assert len(b)== 1000
assert np.allclose(b, test_data["task2"]["b"], atol=1)
print("Ok!")


# In[36]:


draw_signal(test_data['task2']['c'])


# In[37]:



c = a-b
draw_signal(c)


# In[38]:


assert len(c)== 1000
assert np.allclose(c, test_data["task2"]["c"], atol=1)
print("Ok!")


# # Задание 3. Свертка (3 балла)

# ## Теория
# Одна из наиболее частых операций, которая выполняется при обработке сигналов, это свёртка. Свёртка имеет много различных применений, например, с ее помощью можно убрать из сигнала шумы или применить к сигналу эффект эхо.
# 
# 
# Свёртка — это математическая операция, применённая к двум функциям f и g и порождающая третью функцию. Операцию свёртки можно интерпретировать как «схожесть» одной функции с отражённой и сдвинутой копией другой.  Другими словами, преобразование свёртки однозначно определяет выходной сигнал y(t) для установленного значения входного сигнала x(t) при известном значении функции импульсного отклика системы h(t).
# 
# ![Convolution](resources/Convolution_of_box_signal_with_itself2.gif "Convolution")
# 
# Формула свёртки:
# $$y_t=\frac{1}{2} \int_0^T x(\tau)h(t-τ)dτ$$
# где $\tau$  - длительность импульсной переходной характеристики.
# 
# ## Практика
# Реализуйте операцию свёртки. Сравните её с существующей реализацией scipy.signal.convolve. Постройте графики фильтра, исходного сигнала и результата свертки.

# In[27]:


#Операция свертки
        
def convolve(in1, in2):
    P, Q, N = len(in1), len(in2), len(in1) + len(in2) - 1
    result = []
    m=[]
    n=[]
    for k in range(N):
        t, lower, upper = 0, max(0, k - (Q - 1)), min(P - 1, k)
        for i in range(lower, upper + 1):
            t = t + in1[i] * in2[k - i]
        result.append(t)
    return np.array(result) 


# In[28]:


def test_convolve(a, b, print_debug=False):
    my_result = convolve(a, b)
    scipy_result = scipy.signal.convolve(a, b, method='direct')
    if print_debug:
        print(f"Your result {my_result}")
        print(f"Scipy result {scipy_result}")
    assert np.allclose(my_result, scipy_result), f"Test {a} conv {b} failed"
    print("Ok!")


# In[29]:


a = np.repeat([0,1,0], 10)
print(a)
b = np.array([0,1,2,3,2,1,0])
print(b)
myresult=convolve(a,b)

t_response = np.linspace(-10,10,len(a))
t_response_b = np.linspace(-10,10,len(b))
t_response_conv=np.linspace(-10,10,len(myresult))


# In[31]:


# Нарисуйте результат свертки a и b

plt.plot(t_response_conv,myresult, label = 'convolution result')
plt.plot(t_response,a  , label = r' исходный сигнал ')
plt.plot(t_response_b,b, label = r'фильтр')
plt.legend()
plt.show()


# In[1223]:


test_convolve(a, b, print_debug=False)


# # Задание 5. * Алгоритм Карплуса-Стронга 
# 
# Реализуйте  [Алгоритм Карплуса-Стронга](https://en.wikipedia.org/wiki/Karplus%E2%80%93Strong_string_synthesis). В качестве фильтра используйте усреднитель двух смежных отсчетов. Проверьте результат. 
# 
# Отрисуйте и воспроизведите полученный сигнал. На что влияют параметры генерации? Попробуйте имитировать звучание разных струн гитары.

# In[49]:


def karplus_strong(noise, N):
    # Noise - input
    # N - number of samples to generate
    # return y - generated signal based on Noise 
    
    y = np.zeros(int(N))
    for i in range(len(noise)):
        y[i] = noise[i]
    for i in range(len(noise), len(y)):
        y[i] = (y[i-len(noise)]+y[i-len(noise)-1])/2
        y = y / np.max(np.abs(y))  
          
    return np.int16(y * 32767)
    #
    raise NotImplementedError()


# In[50]:


np.random.seed(seed=1)
sample_rate = 44100 
frequency = 82.41
sec = 2
gen_len = sample_rate * sec
noise = (2 * np.random.uniform(-1, 1, int(sample_rate/frequency))) # [-1, 1]

gen_wav = karplus_strong(noise, gen_len)
assert np.allclose(gen_wav[:len(noise)], noise), "Generated signal must starting with noise"
assert np.allclose(gen_wav[len(noise)], (noise[0])/2), "Out of range samples eq 0."
assert np.allclose(gen_wav[len(noise)+1: 2*len(noise)], (noise[:-1] + noise[1:])/2),     "Bad requrent rule( 1 iteration)"
assert np.allclose(gen_wav[2*len(noise)], (noise[0]/2 + noise[-1])/2),     "Bad requrent rule( 2 iteration)"
assert np.allclose(gen_wav[2*len(noise)+2: 3*len(noise)],                    (((noise[:-1] + noise[1:])/2)[:-1] + ((noise[:-1] + noise[1:])/2)[1:])/2),     "Bad requrent rule( 3 iteration)"
print('All Ok!')


# In[51]:


np.random.seed(seed=1)
sample_rate = 44100 
frequency = 82.41
sec = 2
gen_len = sample_rate * sec
noise = (2 * np.random.uniform(-1, 1, int(sample_rate/frequency))) # [-1, 1]

gen_wav = karplus_strong(noise, gen_len)
wavfile.write("./file.wav", sample_rate, gen_wav)
ipd.Audio("./file.wav", autoplay=True)


# In[52]:


sr, x = wavfile.read("C:/Users/Seelowe/Desktop/audio/file.wav")
t = np.linspace(0,x.shape[0]/sr,x.shape[0])

fig, ax=plt.subplots()
ax.plot(t, x, linewidth = 0.05)
ax.set(xlabel= "Time in seconds", ylabel="Amplitude")
plt.title("K-S algorithm ")
plt.show()


# In[53]:


# Попробуйте покрутить параметры генерации. 

np.random.seed(seed=1)
sample_rate =  44100
frequency = 180
sec = 2
gen_len = sample_rate * sec
noise = (2 * np.random.uniform(-1, 1, int(sample_rate/frequency)))

gen_wav = karplus_strong(noise, gen_len)
wavfile.write("./file.wav", sample_rate, gen_wav)
ipd.Audio("./file.wav", autoplay=True)


# In[54]:



sr, x = wavfile.read("./file.wav")
t = np.linspace(0,x.shape[0]/sr,x.shape[0])

fig, ax=plt.subplots()
ax.plot(t, x, linewidth = 0.05)
ax.set(xlabel= "Time in seconds", ylabel="Amplitude")
plt.title("K-S algorithm ")
plt.show()


# In[ ]:





# In[ ]:




