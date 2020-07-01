# Modelos de Deep Learning aplicados a imagens médicas como ferramenta no diagnóstico de COVID-19 
# Deep Learning models applied to medical images as a tool in the diagnosis of COVID-19

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qduTX2yPoMB6Ejhei3-PkB4BkdnqwAbd?usp=sharing)

# Abstract
Different regions of the world have been affected by the new coronavirus. A huge number of people are getting sick and taken to the hospital because one of the symptoms makes the affected person unable to breath. A well accepted and reliable way to detect the virus in patients is the RT-PCR (reverse-transcriptase polymerase chain reaction) test. However, not all nations have access to this form of diagnosis and when it exists, it can be scarce and expensive. Another way to perform the diagnosis of COVID-19 is through Computerized Tomography (CT). Although it is a good alternative to RT-PCR, underdeveloped regions may not have the necessary equipment, and it is a longer procedure than an X-ray examination. X-ray images are an easy-to-access alternative in different regions of the world, and are fast exams, but are less effective in detecting the disease. Therefore, the main motivation of this work is to use machine learning, deep learning to be more specific, to accelerate and increase the accuracy of the detection of COVID-19 through X-ray images, and thus enable its use as a diagnosis.

Different regions of the world have been affected by the new coronavirus. A huge number of people are getting sick and taken to the hospital because one of the symptoms makes the affected person unable to breath. Reliable way to detect the virus are expensive and getting scarce, or don’t even exist in isolated places. To help doctors detect more easily the sickness, people around the world are coming together to develop machine learning algorithms in order to make the diagnosis of the disease faster and more accurate through image analysis, so even in isolated places, they can be diagnosed with a simple computer. Therefore, this work aims to propose an analysis of some deep learning models with  X-ray and computerized tomography images to detect the disease and see if it is possible to further improve models proposed by others. 


# Equipe
* Eduardo Ferreira - R.A. 139407
* Henrique Orpheu - R.A. 139505
* Leandro Carvalho - R.A. 228595
* Nathan Ribeiro - R.A. 263732

# Vídeo de apresentação do projeto
https://www.youtube.com/watch?v=ps3WlKUGLFw&feature=youtu.be

# Introdução
O ano de 2020 será lembrado por inúmeras gerações futuras devido a epidemia que, inesperadamente, se espalhou mundialmente e se tornou uma pandemia. A doença, causada por um coronavírus nomeado COVID-19, desencadeia uma sindrome respiratória aguda grave (SARS), em específico o Sars-CoV-2. Uma parcela da população que contrai o vírus possui quadro assintomático, porém os que desenvolvem sintomas, acabam sofrendo de infecção pulmonar e necessitam de auxilio de respiradores artificiais para sobreviver. Com a evolução da tecnologia, ferramentas computacionais se tornam utensílios de grande valor para o diagnóstico precoce das doenças.

Segundo [(JIANPNEG et al., 2020)](https://www.researchgate.net/publication/340271344_COVID-19_Screening_on_Chest_X-ray_Images_Using_Deep_Learning_based_Anomaly_Detection), deep learning é uma ferramenta efetiva para auxiliar radiologistas na detecção de anomalias em imagens, por isso, nossa proposta é utilizar imagens de raio-x dos pulmões e de tomografia computadorizada para identificar os casos de COVID-19. Para tal, iremos explorar o uso de modelos de deep learning, com o objetivo de identificar qual o modelo que apresenta melhor performance para a tarefa de classificar imagens de exames realizados e prover uma assistência ao diagnóstico médico rápido.

# Motivação

Por se tratar de um vírus que originou uma pandemia, diferentes regiões do mundo foram afetadas. Uma forma com boa aceitação e confiabilidade para detecção do vírus em pacientes é o teste RT-PCR (do inglês *reverse-transcriptase polymerase chain reaction*). Entretando, nem todas as nações tem acesso a essa forma de diagnóstico e quando existente, pode ser escasso.

Uma outra forma de realizar o diagnóstico do COVID-19 é através das Tomografias Computadorizadas (CT). Apesar de ser uma boa alternativa à RT-PCR, regiões sub-desenvolvidas podem não ter o equipamento necessário, além de ser um procedimento mais longo do que um exame de Raio-X.
As imagens de Raio-X são uma alternativa de fácil acesso em diferentes regiões do mundo, e são exames rápidos, porém apresentam uma menor eficácia na detecção da doença.

Dessa forma, a principal motivação desse trabalho é utilizar a aprendizagem de máquina para acelerar e aumentar a acurácia da detecção do COVID-19 através de imagens de Raio-X, e assim viabilizar a sua utilização como diagnóstico.

## Perguntas de Pesquisa
* É possível por meio de aprendizagem de máquina e visão computacional, criação de um modelo capaz de classificar imagens de raio-x de exames de COVID-19?
* Dado o que já existe de modelos de aprendizagem de máquina, é possível melhorar a acurácia dos modelos já existentes com o uso de tecnologias/técnicas diferentes (redes neurais, etc)?

## Objetivos do projeto

Nossa proposta é utilizar imagens de raio-x dos pulmões e de tomografia computadorizada para classificar a partir destas imagens, casos de COVID-19.

Para tal, iremos explorar o uso de modelos de redes neurais, aplicadas para a tarefa de classificação de imagens de exames, para de alguma forma, possívelmente auxiliar no diagnóstico médico rápido para casos de COVID-19.

# Recursos e Métodos

## Bases de Dados
Base de Dados | Endereço na Web | Resumo descritivo e uso
----- | ----- | -----
COVID-CT-Dataset: a CT scan dataset about COVID-19 - Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao (2020) | https://github.com/UCSD-AI4H/COVID-CT | Dataset com 349 imagens de tomografia computadorizada que contém achados clínicos referentes ao diagnóstico positivo para COVID-19.
COVID-19 - China Consortium of Chest CT Image Investigation(CC-CCII) | http://ncov-ai.big.ac.cn/download?lang=en | Dataset com aproximadamente 4000 imagens de exames positivos e negativos para o diagnóstico de COVID-19, dada a quantidade de imagens e tamanho da base nos aproveitamos de uma parcela bem pequena das imagens disponibilizadas.
COVID-19 image data collection - Cohen, Joseph Paul and Morrison, Paul and Dao, Lan | https://github.com/ieee8023/covid-chestxray-dataset | Dataset com 803 imagens de Raio-X e Tomografia Computadorizada distribuidas em diagnósticos positivos e negativos para COVID-19, nos utilizamos das imagens de Raio-X disponíveis nesta base
"Chest X-Ray Images (Pneumonia)" - Mooney, Paul | https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia | Dataset com 5863 imagens de Raio-X de pulmão com diagnósticos normais e para Pneumonia (viral e bacterial), nos utilizamos deste dataset para aumentar a quantidade de arquivos negativos para COVID-19

## Ferramentas

Ferramenta | Endereço na Web | Resumo descritivo e uso
----- | ----- | -----
Google Colab | https://research.google.com/colaboratory/faq.html | Nos utilizamos das ferramentas disponibilizadas pelo Google Colab para a construção do Notebook com as análises detalhadas que fizemos
Orange Data Mining | https://orange.biolab.si/ | Fizemos uma implementação de modelos de classificação na ferramenta Orange Data Mining para avaliar os resultados de modelos simplificados para a solução do problema de classificação das imagens de COVID-19

# Metodologia
~~~
<Abordagem/metodologia adotada, incluindo especificação de quais técnicas foram exploradas, tais como: aprendizagem de máquina, análise de redes, análise estatística, ou integração de uma ou mais técnicas.>
~~~

## Detalhamento do Projeto

### Orange

O Orange Canvas é uma ferramenta de visualização e manipulação de dados de código aberto, também é uma ferramenta muito usada em machine learning, sua interface gráfica é didática a ponto de qualquer pessoa como o mínimo de conhecimento seja capaz de criar e manipular um modelo de visualização ou até mesmo classificação dos dados.

Nós usamos essa ferramenta para comparar com o nosso modelo criado no Google Colab, como o Orange já tem modelos de machine learnig em sua biblioteca acabamos por usar esses modelos já criados.
Nós usamos a base de dados de imagens de raio x com a divisão de 70% das imagens para treinamento e 30% das imagens para o teste, e obtivemos um resultado satisfatório.
 
~~~python
df = pd.read_excel("/content/drive/My Drive/Colab Notebooks/dataset.xlsx");
sns.set(color_codes=True);
sns.distplot(df.Hemoglobin);
plt.show();
~~~

## Evolução do Projeto
~~~
<Relate a evolução do projeto: possíveis problemas enfrentados e possíveis mudanças de trajetória. Relatar o processo para se alcançar os resultados é tão importante quanto os resultados.>
~~~

# Resultados e Discussão
~~~
<Apresente os resultados da forma mais rica possível, com gráficos e tabelas. Mesmo que o seu código rode online em um notebook, copie para esta parte a figura estática. A referência a código e links para execução online pode ser feita aqui ou na seção de detalhamento do projeto (o que for mais pertinente).

A discussão dos resultados também pode ser feita aqui na medida em que os resultados são apresentados ou em seção independente. Aspectos importantes a serem discutidos: É possível tirar conclusões dos resultados? Quais? Há indicações de direções para estudo? São necessários trabalhos mais profundos?>
~~~

# Conclusões
Com base nas análises, foi possível observar a grande dificuldade em trabalhar com imagens para classificação, além de ser uma tarefa que demanda computacionalmente por se tratar de um grande volume de dados, é ainda mais complexa pelas grande quantidade de modelos disponíveis para trabalhar na classificação de imagens.

Foi interessante notar, que resultados satisfatórios para o conjunto de dados que coletamos foram obtidos por meio da aplicação de técnicas de transfer learning, onde certas camadas de uma rede neural são pré-treinadas em um grande conjunto de dados, e somente aplicadas no dataset em análise após a criação de novas camadas de saída facilita na extração de features relevantes para a classificação.

Em particular tivemos grandes dificuldades para lidar com a grande quantidade de arquivos no ambiente de execução do Google Colab, existe limitações na memória disponível na máquina alocada para processar os dados que gerou a necessidade de por exemplo limitar a quantidade de imagens analisadas e também pode ter sido um fator agravante para a acurácia dos modelos avaliados, em particular, para os conjuntos de dados de tomografia.

Outra dificuldade inicial foi a escolha do conjunto de dados que iriamos utilizar para executar as análises, apesar da grande quantidade de imagens disponíveis, muitas delas estavam hospedadas em servidores chineses que tinham taxas de transferência extremamente baixas. Inicialmente foram testadas outras metodologias para a seleção de dados de treino e teste e que gerou uma baixa acurácia dos resultados e testes efetuados, sendo necessário posteriormente alterar as funções de distribuição para ferramentas mais adequadas oferecidadas por bibliotecas como sklearn e Tensorflow.

Foi interessante observar que camadas convolucionais, comumente usadas na classsificação de imagens convencionais (animais, objetos), tiveram um bom desempenho no conjunto de dados médicos analisados, mesmo que, em arquiteturas simplificadas.

Finalmente, considerando o conjunto de imagens de tomografia computadorizada, chegamos a conclusão de que utilizar métodos de classificação podem não ser as ferramentas mais adequadas para auxiliar no diagnóstico de COVID-19, dado que mesmo as arquiteturas de redes neurais mais complexas apresentarem difulculdades para extrair features das imagens de tomografia computadorizada.

# Trabalhos Futuros

Considerando os trabalhos futuros, poderiamos explorar o ajuste e parâmetros dos modelos treinados, assim como a modificação das camadas inferiores e de saída dos modelos com o objetivo de melhorar os resultados obtidos para as imagens de tomografia. Outra possibilidade seria explorar métodos de segmentação com redes neurais que também ajudariam no objetivo de identificar regiões de atenção nos exames e que seriam ferramentas visuais para o auxílio de médicos no diagnóstico da COVID-19.


# Referências

[HE, X. et al. Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans.medrxiv, 2020.](https://github.com/UCSD-AI4H/COVID-CT)

[Kang Zhang, Xiaohong Liu, Jun Shen, et al. Jianxing He, Tianxin Lin, Weimin Li, Guangyu Wang. (2020). Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography.](http://ncov-ai.big.ac.cn/download?lang=en)

[Paul Cohen, Morrison, Dao, et al. COVID-19 Image Data Collection: Prospective Predictions Are the Future, arXiv:2006.11988, 2020 ](https://github.com/ieee8023/covid-chestxray-dataset)

[Kermany, Daniel S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." Cell 172.5 (2018): 1122-1131.](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)


JIANPENG, Z. et al. COVID-19 Screening on Chest X-ray Images Using Deep Learning basedAnomaly Detection, 2020

