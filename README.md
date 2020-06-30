# Estrutura de Arquivos e Pastas

A estrutura aqui apresentada é uma simplificação daquela proposta pelo [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

~~~
├── README.md          <- Apresentação do projeto
│
├── data
│   └── external       <- Dados de terceiros
│
├── notebooks          <- Notebooks exportados do Google Colab
│
├── src                <- fonte em linguagem de programação ou sistema (e.g., Orange)
│   └── README.md      <- instruções básicas de instalação/execução
│
└── assets             <- mídias usadas no projeto
~~~

Na raiz deve haver um arquivo de nome `README.md` contendo a apresentação do projeto, como detalhado na seção seguinte.

## `data`

Dados utilizados no projeto respeitadas as possíveis implicações éticas, se você tiver licença para tal e se o volume for suportado pelo Github. Você pode optar por colocar um subconjunto ilustrativo dos dados.

É importante que sejam colocados os dados originais (se for possível) para garantir a reprodutibilidade do processo. Os originais são colocados na subpasta `raw` se forem produzidos pela equipe e na subpasta `external` se forem de terceiros. Também podem ser colocados aqui dados intermediários (por exemplo, dados tratados, resumidos etc.) na pasta `interim`. Finalmente, coloque os dados finais que serviram de entrada para as suas análises na subpasta `processed`.

## `notebooks`

Código do seu projeto que pode ser executado online sem instalação de software, tal como um notebook em Jupyter ou equivalente.

## `src`

Código em alguma linguagem ou projeto em Orange, Weka e similares.

Se for código em linguagem de programação, tente organizá-lo de forma que seja simples a sua execução por terceiros, por exemplo, acrescente as bibliotecas necessárias etc. Acrescente na raiz um arquivo `README.md` com as instruções básicas de instalação e execução.

## `assets`

Qualquer mídia usada no seu projeto: vídeo, ilustrações, arquivos PDF etc.

# Modelos de Deep Learning aplicados a imagens médicas como ferramenta no diagnóstico de COVID-19 
# Deep Learning models applied to medical images as a tool in the diagnosis of COVID-19

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qduTX2yPoMB6Ejhei3-PkB4BkdnqwAbd?usp=sharing]

# Introdução
O ano de 2020 será lembrado por inúmeras gerações futuras devido a epidemia que, inesperadamente, se espalhou mundialmente e se tornou uma pandemia. A doença, causada por um coronavírus nomeado COVID-19, desencadeia uma sindrome respiratória aguda grave (SARS), em específico o Sars-CoV-2. Uma parcela da população que contrai o vírus possui quadro assintomático, porém os que desenvolvem sintomas, acabam sofrendo de infecção pulmonar e necessitam de auxilio de respiradores artificiais para sobreviver. Com a evolução da tecnologia, ferramentas computacionais se tornam utensílios de grande valor para o diagnóstico precoce das doenças.

Segundo [(JIANPNEG et al., 2020)](https://www.researchgate.net/publication/340271344_COVID-19_Screening_on_Chest_X-ray_Images_Using_Deep_Learning_based_Anomaly_Detection), deep learning é uma ferramenta efetiva para auxiliar radiologistas na detecção de anomalias em imagens, por isso, nossa proposta é utilizar imagens de raio-x dos pulmões e de tomografia computadorizada para identificar os casos de COVID-19. Para tal, iremos explorar o uso de modelos de deep learning, com o objetivo de identificar qual o modelo que apresenta melhor performance para a tarefa de classificar imagens de exames realizados e prover uma assistência ao diagnóstico médico rápido.

A estrutura deste trabalho é divida como apresentado a seguir. A seção metodologia introduz a fundamentação teórica utilizada em nosso trabalho, assim como descreve a comparação dos métodos de deep learning utilizados na pesquisa. Na seção materiais apresentamos os materiais e ferramentas utilizados para a construção do sistema de coleta de dados necessários para realizar as análises desejadas. Na seção análises apresentamos as características dos resultados coletados assim como efetuamos as análises realizadas sob os mesmos. Finalmente, na seção conclusões, concluí-se o trabalho apresentando as perspectivas gerais dos resultados e explora as melhorias e expansões dos experimentos em trabalhos futuros.

# Abstract in English
~~~
<English version of the abstract.>
~~~

# Equipe
* Eduardo Ferreira - R.A.
* Henrique Orpheu - R.A.
* Leandro Carvalho - R.A.
* Nathan Ribeiro - R.A. 263732

# Vídeo de apresentação do projeto
https://www.youtube.com/watch?v=ps3WlKUGLFw&feature=youtu.be

# Motivação

Por se tratar de um vírus que originou uma pandemia, diferentes regiões do mundo foram afetadas. Uma forma com boa aceitação e confiabilidade para detecção do vírus em pacientes é o teste RT-PCR (do inglês *reverse-transcriptase polymerase chain reaction*). Entretando, nem todas as nações tem acesso a essa forma de diagnóstico e quando existente, pode ser escasso.

Uma outra forma de realizar o diagnóstico do COVID-19 é através das Tomografias Computadorizadas (CT). Apesar de ser uma boa alternativa à RT-PCR, regiões sub-desenvolvidas podem não ter o equipamento necessário, além de ser um procedimento mais longo do que um exame de Raio-X.
As imagens de Raio-X são uma alternativa de fácil acesso em diferentes regiões do mundo, e são exames rápidos, porém apresentam uma menor eficácia na detecção da doença.

Dessa forma, a principal motivação desse trabalho é utilizar a aprendizagem de máquina para acelerar e aumentar a acurácia da detecção do COVID-19 através de imagens de Raio-X, e assim viabilizar a sua utilização como diagnóstico.

## Perguntas de Pesquisa
~~~
<Perguntas de pesquisa que o projeto pretende responder ou hipóteses a serem avaliadas, enunciadas de maneira objetiva e verificável.>
~~~

## Objetivos do projeto
~~~
<Como seu projeto propôs abordar o problema apresentado.>
~~~

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
~~~
<Apresente aqui detalhes da análise. Nesta seção ou na seção de Resultados podem aparecer destaques de código como indicado a seguir. Note que foi usada uma técnica de highlight de código, que envolve colocar o nome da linguagem na abertura de um trecho com `~~~`, tal como `~~~python`.

Os destaques de código devem ser trechos pequenos de poucas linhas, que estejam diretamente ligados a alguma explicação. Não utilize trechos extensos de código. Se algum código funcionar online (tal como um Jupyter Notebook), aqui pode haver links. No caso do Jupyter, preferencialmente para o Binder abrindo diretamente o notebook em questão.>
~~~

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
~~~
<Apresente aqui as conclusões finais do trabalho e as lições aprendidas.>
~~~

# Trabalhos Futuros
~~~
<Indique trabalhos futuros a partir do ponto alcançado.>
~~~