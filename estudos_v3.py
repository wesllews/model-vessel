from operator import length_hint
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from skimage.draw import line
from skimage import data, draw
from skimage.draw import circle_perimeter
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage, stats
import time
import inspect

def geraPonto(grafo, edge):
    """
    INPUT
    -----
    grafo : dict, networkx.DiGraph() que tem na arestas os valores de: 1- Angulo global, o angulo que o seguimento anterior tinha até chegar naquele ponto; 2- Comprimento, tamanho do segmento que o vetor vai direcionar, item sorteado na função atributos2; 3- pontoReal, são as coordenadas em valores decimais, ou seja, eixo x,y dos pontos; 
    
    edge : tuple, contém a aresta que será criado o pronto da posição edge[1] 
    
    
    OUTPUT
    -----
    pontoReal : tuple, contendo valores de x e y do ponto em unidades reais (eixo cartesiano)
    """
    theta = np.radians(grafo.edges[edge]["anguloGlobal"])
    versor = np.array((np.cos(theta), np.sin(theta))) # Definir o versor, ou seja a direção da reta

    # Definir o vetor;
    vetor = grafo.edges[edge]["comprimento"]*versor    # Vetor representando a reta

    #Definir o ponto
    pontoReal = (grafo.nodes[edge[0]]["pontoReal"] + vetor)   # Deslocamos o ponto inicial do vetor
    return pontoReal

def cordenadasMinimosMaximos(grafo: dict):
    """
    INPUT
    -----
    grafo : dict, networkx.DiGraph() que tem os nós atuais do grafo e a característica "pontoReal" que são as coordenadas dos nós em valores decimais, ou seja, eixo x,y dos pontos; 

    OUTPUT
    -----
    xmin, ymin: float, contém o menor valor encontrado para x e y respectivamente dentro de pontos
    pontos : list, contém uma matriz em que cada index contém uma lista com as cordenadas x,y de um nó do grafo.
    """
    nodes = list(grafo.nodes())
    dictPontos = {}
    pontos = []

    # PROBLEMA: Teste para ver aonde tem nós sendo retirados da lista ou temos divergencia na sequencia de nós
    for node in nodes:
        if "pontoReal" in grafo.nodes[node]:
            dictPontos[node] = grafo.nodes[node]["pontoReal"]
            pontos.append(grafo.nodes[node]["pontoReal"])
    x, y = zip(*pontos)
    xmin = min(x)
    ymin = min(y)
    
    return xmin, ymin, dictPontos

def convertePontoParaPixel(grafo, pontoInicial: list, resolucao = 0.1):
    """
    INPUT
    -----
    grafo : dict, networkx.DiGraph() que tem os nós atuais do grafo e a característica "pontoReal" que são as coordenadas dos nós em valores decimais, ou seja, eixo x,y dos pontos; 
    
    pontoInicial : list(x,y), contém as cordenadas decimais do ponto que se espera projetar em pixel 
    
    resolucao : float, A resolução de uma imagem é o número de pontos por pixel em uma imagem


    OUTPUT
    -----
    pontoReal : tuple, contendo valores de x e y do ponto em unidades reais (eixo cartesiano)
    """
    xmin, ymin, semUso_dictPontos = cordenadasMinimosMaximos(grafo)
    
    limite = -15
    xmin= min([xmin, limite])
    ymin= min([ymin, limite])


    x = pontoInicial[0]
    y = pontoInicial[1]
    x_new = x - xmin
    y_new = y - ymin
        
    x_final = int(x_new/resolucao)
    y_final = int(y_new/resolucao)

    return [x_final,y_final]

def cordenadasParaPixel(grafo, resolucao = 0.1):
    """
    INPUT
    -----
    grafo : dict, networkx.DiGraph() que tem os nós atuais do grafo e a característica "pontoReal" que são as coordenadas dos nós em valores decimais, ou seja, eixo x,y dos pontos; 

    resolucao : float, A resolução de uma imagem é o número de pontos por pixel em uma imagem


    OUTPUT
    -----
    pixel : list, lista de tuplas contendo cordenadas em pixel de todos os nodes do grafo, sendo a posição no array o número do node.
    """
    xmin, ymin, dictPontos = cordenadasMinimosMaximos(grafo)
    
    limite = -15 # proximo; isso pode gerar artefatos nas bordas
    xmin= min([xmin, limite])
    ymin= min([ymin, limite])

    pixel = []
    dictPixel = {}
    nodes = list(grafo.nodes())
    for node in nodes:
        if "pontoReal" in grafo.nodes[node]:
            x_new = dictPontos[node][0] - xmin
            y_new = dictPontos[node][1] - ymin
            
            x_final = int(x_new/resolucao)
            y_final = int(y_new/resolucao)
                    
            pixel.append((x_final,y_final))
            dictPixel[node] = (x_final,y_final)
    return pixel, dictPixel

def projetaCampo(grafo:dict, resolucao:float = 0.1, raio:float = 1, margem = 20):
    """
    INPUT
    -----
    grafo : dict, networkx.DiGraph() contendo no minimo uma aresta inicial com todos os atributos preenchidos
    
    resolucao : float, A resolução de uma imagem é o número de pontos por pixel em uma imagem
    
    raio : float, comprimento/raio do circulo que será projetado em torno do ponto selecionado (rowPonto,colPonto)
    
    padding : int, preenchimento interno na imagem de forma que o circulo projetado a partir do raio não tenha 'index out range'
       
    
    OUTPUT
    -----
    vasosSanguineos : np.array(), matriz numpy  contendo as posições em pixel dos vasos sanguíneos contidos no grafo recebido
    
    campoProjetado : np.array(), matriz contendo os valores da transformada distancia entre um pixel do plano e um vaso sanguíneo já contido no grafo
    """
    
    # Define um paddin com base no tamanho de raio
    padding = raio+20
        
    # Converte as coordenadas Reais dos pontos presentes no grafo até então para valores de Pixel
    pixels, dictPixels = cordenadasParaPixel(grafo, resolucao = resolucao)
    pixels = np.asarray(pixels)

    # Pega os maiores valores de colunas(x) e linhas(y)
    colunas = max(pixels[:,0])+1 # Todos valores na posição 0(x) das listas de pontos
    linhas = max(pixels[:,1])+1  # Todos valores na posição y(y) das listas de pontos
    
    # Define uma matriz inicial com os valores de tamanhos máximos pra linha e colunas, adicionando espaço de padding nos 4 lados da matriz
    vasosSanguineos = np.zeros(shape = (linhas+2*padding+1, colunas+2*padding+1), dtype=np.uint8)

    for edge in grafo.edges():
        if "pontoReal" in grafo.nodes[edge[1]]: # Interrompe o loop caso não haja atribuição de pontos para os edges já definidos
            x1,y1 = dictPixels[edge[0]]
            x2,y2 = dictPixels[edge[1]]
            rr,cc = line(int(y1)+padding,int(x1)+padding,int(y2)+padding,int(x2)+padding)
            vasosSanguineos[rr,cc] = 1

    """ # Adicionando uma linha em cada uma das quatro bordas da MATRIZ usada para projetar o campo
             linhas = linhas+2*padding+1
             colunas = colunas+2*padding+1
             vasosSanguineos[:,margem-5:margem+6] = 1                    # Vertical esquerda
             vasosSanguineos[:,colunas-margem-5: colunas-margem+6] = 1   # Vertical Direita
             vasosSanguineos[margem-5:margem+6,:] = 1                    # Horizontal superior
             vasosSanguineos[linhas-margem-5:linhas-margem+6,:] = 1      # Horizontal superior"""

    # Fazendo o campo de repulsão (transformada de distância)
    from scipy import ndimage
    campoProjetado = ndimage.distance_transform_edt(np.logical_not(vasosSanguineos), indices=np.int32) #inverter pq quando se tem apenas uma linha você uma proximidade muito grande do zero e acaba tendo um resultado nao significativo

    return vasosSanguineos, campoProjetado

def projetaCampoContinuacao(grafo:dict, resolucao:float = 0.1, raio:float = 1, margem = 20):
    """
    INPUT
    -----
    grafo : dict, networkx.DiGraph() contendo no minimo uma aresta inicial com todos os atributos preenchidos
    
    resolucao : float, A resolução de uma imagem é o número de pontos por pixel em uma imagem
    
    raio : float, comprimento/raio do circulo que será projetado em torno do ponto selecionado (rowPonto,colPonto)
    
    padding : int, preenchimento interno na imagem de forma que o circulo projetado a partir do raio não tenha 'index out range'
       
    
    OUTPUT
    -----
    vasosSanguineos : np.array(), matriz numpy  contendo as posições em pixel dos vasos sanguíneos contidos no grafo recebido
    
    campoProjetado : np.array(), matriz contendo os valores da transformada distancia entre um pixel do plano e um vaso sanguíneo já contido no grafo
    """
    
    # Define um paddin com base no tamanho de raio
    padding = 0
        
    # Converte as coordenadas Reais dos pontos presentes no grafo até então para valores de Pixel
    pixels, dictPixels = cordenadasParaPixel(grafo, resolucao = resolucao)
    pixels = np.asarray(pixels)
    
    # Pega os maiores valores de colunas(x) e linhas(y)
    colunas = max(pixels[:,0])+1
    linhas = max(pixels[:,1])+1
    
    # Define uma matriz inicial com os valores de tamanhos máximos pra linha e colunas, adicionando espaço de padding nos 4 lados da matriz
    vasosSanguineos = np.zeros(shape = (linhas, colunas), dtype=np.uint8)

    for edge in grafo.edges():
        if "pontoReal" in grafo.nodes[edge[1]]: # Interrompe o loop caso não haja atribuição de pontos para os edges já definidos
            x1,y1 = dictPixels[edge[0]]
            x2,y2 = dictPixels[edge[1]]
            rr,cc = line(int(y1),int(x1),int(y2),int(x2))
            vasosSanguineos[rr,cc] = 1

    # Adicionando uma linha em cada uma das quatro bordas da MATRIZ usada para projetar o campo
    vasosSanguineos[:,0:60] = 1                 # Vertical esquerda
    vasosSanguineos[:,colunas-60:colunas] = 1   # Vertical Direita
    vasosSanguineos[0:60,:] = 1                 # Horizontal superior
    vasosSanguineos[linhas-60:linhas,:] = 1     # Horizontal superior

    # Fazendo o campo de repulsão (transformada de distância)
    from scipy import ndimage
    campoProjetado = ndimage.distance_transform_edt(np.logical_not(vasosSanguineos), indices=np.int32) #inverter pq quando se tem apenas uma linha você uma proximidade muito grande do zero e acaba tendo um resultado nao significativo

    return vasosSanguineos, campoProjetado

def retornaIntensidades(img: list, rowPonto: int, colPonto: int, raio:float, grauInclinacao: int = 3, valorCentral: int = 4.8, padding = 50, modo:str = "disco"):
    """
    INPUT
    -----
    img : list, matriz de pixel com transformação dos campos aplicada
    
    rowPonto : int, linha do pixel selecionado
    
    colPonto : int, coluna do pixel selecionado
    
    raio : float, comprimento/raio do circulo que será projetado em torno do ponto selecionado (rowPonto,colPonto)
    
    grauInclinacao : int, determina o quanto a função apresenta comportamentos retilíneos, ou seja, uma função saltada ou uma função curvilinea sem grandes saltos
    
    valorCentral : int, determina o ponto crítico da binarização da função
    
    padding: int, valor de margem para gerar as imagens de forma que o comprimento do raio não ultrapasse os index da matriz
    
    modo : define como será contabilizado as intensidades, podendo ser: "perimetro" que utiliza os valores apenas da circunferência; "disco" utiliza o valor médio das intensidades no caminho ponto-circunferência
    
    
    OUTPUT
    -----
    rr_ord : list, indexes ordenados dos valores de linhas do circulo projetado em torno do ponto (rowPonto,colPonto)
    
    cc_ord : list, indexes ordenados dos valores de colunas do circulo projetado em torno do ponto (rowPonto,colPonto)
    
    intensidades : valores das intensidades(0-1) que representam as distâncias entre o ponto e outros ramos já existentes; Esse valor passa por uma transformação sigmoide para diminuir a chance de regiões no sentido de algum ramo existente aconteça
    """
    
    #Desenhando circulo entorno do ponto
    rowPonto = rowPonto+padding
    colPonto = colPonto+padding
    rr, cc = circle_perimeter(rowPonto, colPonto, raio)
    
    # Descobrindo angulos(Radianos) e ordenando por index
    angulos = np.arctan2(rr-rowPonto, cc-colPonto)
    ind = np.argsort(angulos) # Os angulos caminham de 0º -> 180º da direita pra esquerda inferior e de 0 -> -179º da direita pra esquerda superior
    
    # Valores indexados
    rr_ord = rr[ind]
    cc_ord = cc[ind]
    angulos = np.degrees(angulos[ind])
    
    
    if modo == "perimetro":
        # As intensidades serão adquiridas com base no perímetro da circunferência formada em torno do ponto
        intensidades = img[rr_ord, cc_ord]
    
    elif modo == "disco":
        # Usando a função line vamos traçar o caminho dos pixels entre o ponto(rowPonto,colPonto) até o ponto no perímetro do circulo, fazendo a média entre eles e atritbuindo às intensidades
        intensidades = []
        for i in  range(0, len(rr)):
            # Traçando caminho de do ponto até o perimetro
            rr_Aux,cc_Aux = line(rowPonto, colPonto, rr_ord[i], cc_ord[i])
            # Fazendo a média dos pontos
            soma = np.mean(img[rr_Aux, cc_Aux])
            intensidades.append(soma)
    # Tranformação sigmoide
    grauInclinacao = 3
    valorCentral = 4.8
    intensidades = np.array(intensidades)
    intensidadesSigmoide = 1/(1+np.exp(-grauInclinacao*(intensidades-valorCentral)))
    
    return rr_ord, cc_ord, intensidades, intensidadesSigmoide, angulos

def retornaIndex(intensidades: list, threshold:float = 2 ):
    """
    INPUT
    -----
    intensidades : list, matriz contendo valores retornados da retornaIntensidades()
    
    OUTPUT
    -----
    index : int, retorna o único index sorteado item de uma lista contendo todas as intensidades que compõem o circulo entorno do ponto 
    """
    grauInclinacao = 3
    valorCentral = 4.8
    intensidadesSigmoide = 1/(1+np.exp(-grauInclinacao*(intensidades-valorCentral)))
    probabilidades = intensidadesSigmoide/np.sum(intensidadesSigmoide)
    index = np.random.choice(range(len(probabilidades)), size=1, p=probabilidades)
    
    if intensidades[index]<threshold:
        index = "Interrupcao"
    else:
        index = int(index)    
    return index

def angulo180(angulo):
    """
    INPUT
    -----
    angulo : float, valor total ou somatória dos angulos que podem ser maiores que 180º ou menores que -180    
    
    OUTPUT
    -----
    angulo : Angulo ajustado no intervalo [-180,180], sendo valores positivos no sentido anti-horário do zero(horizontal direita) e valores negativos saindo de zero até +-180º no sentido horário
    
    BACKGROUND
    -----
    Os valores de angulos retornados pela função retornaIntensidades é passado de radianos para graus mas são formulados em 0-180 pra os valores da primeira metado do circulo trigonométrico e depos de 0-(-180) na parte inferior mas ambos partem da direita pra esquerda, dessa forma se faz necessária alguns ajustes 
    """
    
    if angulo < -180:
        angulo = 360 + angulo
    elif angulo > 180:
        angulo = - (360 - angulo)

    return angulo

def valoresDoCampo(grafo:dict, predecessorEdge:tuple, edge:tuple, funcaoDoCampo = projetaCampo, resolucao: float = 0.1, mostraGraficos = False):
     #Pega o Ponto inicial em coordenadas reais
    pontoRealInicial = grafo.nodes[predecessorEdge[1]]["pontoReal"]

    #Converte o ponto para linha e coluna de uma matriz (pixel)
    pixelInicial = convertePontoParaPixel(grafo, pontoRealInicial, resolucao = resolucao)

    #Usa o comprimento do segmento como raio do circulo em torno do ponto
    raio = grafo.edges[edge]["comprimento"]
    raioInt = int(raio/resolucao)+1

    # Retorna uma matriz com o campo de repulsão projetado
    img, campo = funcaoDoCampo(grafo, resolucao = resolucao, raio = raioInt)

    # Retorna as cordenadas para o circulo(rr,cc), valores da intensidade e angulo
    rr, cc, intensidades, intensidadesSigmoide, angulos = retornaIntensidades(campo.copy(), rowPonto = pixelInicial[1], colPonto = pixelInicial[0], raio = raioInt-1, padding = raioInt+20) # Quando altera uma lista/matriz em uma função ele altera também fora dela
    
    if mostraGraficos == True:
        lastNode = int(list(grafo.nodes())[-1])
        if lastNode%100 == 0: # Plota a cada 50 nós
            plt.figure()
            plt.suptitle(f'Campo de intensidade, Nº de Nós = {lastNode})', fontsize=14)
            plt.legend(loc='upper left')
            plt.imshow(campo, cmap="gray",vmin=0,vmax=255, origin = "lower")
    return intensidades, intensidadesSigmoide, angulos 

def angulosMaximosMinimos(grafo: dict, predecessorEdge: tuple, intervalo: int):
    """
    INPUT
    -----
    grafo : dict, networkx.DiGraph() que nesse caso precisa dos valores de: 1- Angulo global, o angulo que o seguimento anterior tinha até chegar no edge de interesse;
    
    intervalo : int, valor de diferença acima e abaixo do anguloGlobal, formando uma abertura possível entre máximo e mínimo de 2*intervalo
    
    OUTPUT
    -----
    minimo_aux : Subtração do intervalo no angulo global, sendo o menor angulo possível que este seguimento pode chegar dada sua geometria
    
    minimo : Subtração do intervalo no angulo global assim como o minimo_aux porém com o angulo ajustado no intervalo [-180,180], sendo valores positivos no sentido anti-horário do angulo zero(que fica na horizontal direita) e valores negativos saindo de zero até -180º(que fica na horizontal esquerda) no sentido horário
    
    maximo_aux : Soma do intervalo no angulo global, sendo o maior angulo possível que este seguimento pode chegar dada sua geometria
    
    maximo : Soma do intervalo no angulo global assim como o maximo_aux porém com o angulo ajustado no intervalo [-180,180], sendo valores positivos no sentido anti-horário do angulo zero(que fica na horizontal direita) e valores negativos saindo de zero até -180º(que fica na horizontal esquerda) no sentido horário
    
    
    BACKGROUND
    -----
    Os valores de angulos retornados pela função irão limitar um range de angulos que são possíveis de serem selecionados na circunferência de intensidades, de forma que mesmo sorteados pela intensidades eles respeitem a geometria escolhida pro vaso
    """    
    
    # Recebe o angulo global predecessor
    anguloGlobalpred = grafo.edges[predecessorEdge]["anguloGlobal"]
    
    # Com base no intervalo dado define valor maximo que o angulo pode chegar e o mínimo que pode chegar partirndo do global anterior
    minimo_aux = anguloGlobalpred-intervalo
    maximo_aux = anguloGlobalpred+intervalo
    
   # Usando função angulo180() para adeqquar os valores dentro do espectro de 180 ou -180
    minimo =  angulo180(anguloGlobalpred-intervalo)
    maximo =  angulo180(anguloGlobalpred+intervalo)
    
    return minimo_aux, minimo, maximo_aux, maximo

def retilineo(grafo:dict, predecessorEdge:tuple, edge:tuple, funcaoDoCampo, resolucao: float = 0.1, intervalo = 16):
    # Realiza todo o processo de gerar um campo de repulsão, pegar os valores em torno do ponto
    intensidades, intensidadesSigmoide, angulos = valoresDoCampo(grafo, predecessorEdge, edge, funcaoDoCampo, resolucao)

    # Realiza as contas de valores máximos e minimos com base no intervalo dado e na escala 180º
    minimo_aux, minimo, maximo_aux, maximo = angulosMaximosMinimos(grafo, predecessorEdge, intervalo)
    
    # Selecionando indexes dos angulos e verificando se os angulos ultrapassam os valores críticos
    indexes = []
    if minimo == minimo_aux and maximo == maximo_aux:
        indexes = np.where((angulos > minimo) & (angulos < maximo))

    elif minimo != minimo_aux: # caso o máximo cruse o limite de -180
        indexes = np.where((angulos > minimo) | (angulos < maximo))

    elif maximo != maximo_aux: # caso o máximo cruse o limite de 180
        indexes = np.where((angulos > minimo) | (angulos < maximo))   

    # Recortando os angulos possiveis dentro do +intervalo ou -intervalo
    intensidadesSigmoide, angulos = intensidadesSigmoide[indexes], angulos[indexes]
    index = retornaIndex(intensidades[indexes])
    
    # Interrupção
    flag = 0
    if index == "Interrupcao": 
        grafo.edges[predecessorEdge]["tipo"] = "Interrupcao"
        grafo.remove_edge(*edge)
        grafo.remove_node(edge[1]) # Remove só o último node pq o primeiro está sendo usado no edge predecessor
        flag = "Interrupcao"
    else:
        thetaCampo = angulos[index]

        # Angulo Local (Valor nagativo = crescimento no sentido horário; Valor positivo = cresciemento anti-horário)
        grafo.edges[edge]["anguloLocal"] = angulo180(thetaCampo - grafo.edges[predecessorEdge]["anguloGlobal"]) 

        # Angulo Global
        grafo.edges[edge]["anguloGlobal"] = angulo180(grafo.edges[predecessorEdge]["anguloGlobal"] + grafo.edges[edge]["anguloLocal"])

        # Ponto Real
        grafo.nodes[edge[1]]["pontoReal"] = geraPonto(grafo, edge)
        
        flag = "Continuidade"
        
    return grafo, flag  

def bifurcacaoY(grafo:dict, predecessorEdge:tuple, edge:tuple, funcaoDoCampo, resolucao: float = 0.1, intervalo = 40, modo = "repulsao"):

    # Realiza todo o processo de gerar um campo de repulsão, pegar os valores em torno do ponto
    intensidades, intensidadesSigmoide, angulos = valoresDoCampo(grafo, predecessorEdge, edge, funcaoDoCampo, resolucao)
    
    # Realiza as contas de valores máximos e minimos com base no intervalo dado e na escala 180º
    minimo_aux, minimo, maximo_aux, maximo = angulosMaximosMinimos(grafo, predecessorEdge, intervalo)

    # Recebe o angulo global predecessor
    anguloGlobalpred = grafo.edges[predecessorEdge]["anguloGlobal"]
    
    # Selecionando indexes dos angulos e verificando se os angulos ultrapassam os valores críticos
    indexes = []
    if minimo == minimo_aux and grafo.edges[edge]["tipo"] == "bif_Y_Esquerda":
        indexes = np.where((angulos > minimo) & (angulos < anguloGlobalpred))

    elif maximo == maximo_aux and grafo.edges[edge]["tipo"] == "bif_Y_Direita":
        indexes = np.where((angulos > anguloGlobalpred) & (angulos < maximo))

    elif minimo != minimo_aux: # caso o máximo cruse o limite de -180
        indexes = np.where((angulos > minimo) | (angulos < anguloGlobalpred))

    elif maximo != maximo_aux: # caso o máximo cruse o limite de 180
        indexes = np.where((angulos > anguloGlobalpred) | (angulos < maximo))   

    # Recortando os angulos possiveis +-15
    intensidadesSigmoide, angulos = intensidadesSigmoide[indexes], angulos[indexes]
    # O Teste abaixo garante que se a bifurcação esquerda foi enviada para ser adicionada ela aconteça independente da proximidade com a bif_Y_Direita
    if grafo.edges[edge]["tipo"] == "bif_Y_Esquerda":
        index = retornaIndex(intensidades[indexes], threshold = 0.0)
    else:
        index = retornaIndex(intensidades[indexes])
    
    # Interrupção
    flag = 0
    if index == "Interrupcao": 
        grafo.edges[predecessorEdge]["tipo"] = "Interrupcao"
        grafo.remove_edge(*edge)
        grafo.remove_node(edge[1]) # Remove só o último node pq o primeiro está sendo usado no edge predecessor
        flag = "Interrupcao"
        
    else:
        thetaCampo = angulos[index]

        # Angulo Local
        grafo.edges[edge]["anguloLocal"] = thetaCampo - grafo.edges[predecessorEdge]["anguloGlobal"]

        # Angulo Global
        grafo.edges[edge]["anguloGlobal"] = grafo.edges[predecessorEdge]["anguloGlobal"] + grafo.edges[edge]["anguloLocal"]

        # Ponto Real
        grafo.nodes[edge[1]]["pontoReal"] = geraPonto(grafo, edge)
        
        flag = "Continuidade"
        
    return grafo, flag

def campoDeRepulsão(grafo:dict, predecessorEdge: tuple, edge: tuple, funcaoDoCampo, resolucao: float = 0.1):

     # Realiza todo o processo de gerar um campo de repulsão, pegar os valores em torno do ponto
    intensidades, intensidadesSigmoide, angulos = valoresDoCampo(grafo, predecessorEdge, edge, funcaoDoCampo, resolucao)
    
    # Seleciona o angulo com base na intensidade
    index = retornaIndex(intensidades)

    # Interrupção
    flag = 0
    if index == "Interrupcao": 
        grafo.edges[predecessorEdge]["tipo"] = "Interrupcao"
        grafo.remove_edge(*edge)
        grafo.remove_node(edge[1]) # Remove só o último node pq o primeiro está sendo usado no edge predecessor
        flag = "Interrupcao"
    else:
        thetaCampo = angulos[index]

        # Angulo Local
        grafo.edges[edge]["anguloLocal"] = thetaCampo - grafo.edges[predecessorEdge]["anguloGlobal"]

        # Angulo Global
        grafo.edges[edge]["anguloGlobal"] = grafo.edges[predecessorEdge]["anguloGlobal"] + grafo.edges[edge]["anguloLocal"]

        # Ponto Real
        grafo.nodes[edge[1]]["pontoReal"] = geraPonto(grafo, edge)
        
        flag = "Continuidade"

    return grafo, flag

def plotaGrafo(grafo, res = 0.1, padding = 10):
    """
    Vermelho - Repulsão
    Verde    - Continuação, terminação
    Amarelo  - Lados do Y
    Azul     - Bifurcação em T
    """
    #Definindo matriz no shape = (X, Y)
    pixels, dictPixels = cordenadasParaPixel(grafo, resolucao = res)
    pixels = np.asarray(pixels)

    linhas = max(pixels[:,1])+1
    colunas = max(pixels[:,0])+1
    matriz = np.zeros(shape= (linhas+(2*padding),colunas+(2*padding),3), dtype=np.uint8)

    #print("Edges: ", len(list(grafo.edges())))
    #print("Pixels: ", len(pixels))
    #print(grafo.edges())


    for edge in grafo.edges():
        x1,y1 = dictPixels[edge[0]]
        x2,y2 = dictPixels[edge[1]]
        rr,cc = line(int(y1)+padding,int(x1)+padding,int(y2)+padding,int(x2)+padding)
        if grafo.edges[edge]["tipo"] in ["campoRepulsao"]:
            matriz[rr,cc] = [255, 0, 0]
        elif grafo.edges[edge]["tipo"] in ["Continuacao"]:
            matriz[rr,cc] = [0, 255, 0]
        elif grafo.edges[edge]["tipo"] in ["Terminacao"]:
            matriz[rr,cc] = [29, 143, 177]
        elif grafo.edges[edge]["tipo"] in ["Interrupcao"]:
            matriz[rr,cc] = [255, 105, 180]
        elif grafo.edges[edge]["tipo"] in ["bif_Y_Esquerda","bif_Y_Direita"]:
            matriz[rr,cc] = [255, 255, 0]
        elif grafo.edges[edge]["tipo"] in ["bif_T"]:
            matriz[rr,cc] = [0, 0, 255]
        else:
            print("Edge de Inicio ",edge,":\t",rr[0],",",cc[0])
            matriz[rr,cc] = [255, 255, 255]
            
    """x0,y0 = pixels[0]+padding
    matriz[int(y0-15//res), :] = [255,255,0]
    matriz[:, int(x0-15//res)] = [255,0,0]
    matriz[int(y0+15//res), :] = [255,0,0]
    matriz[:, int(x0+15//res)] = [255,0,0]"""
    return matriz

def comprimentoRamo(comp_min, comp_max):
    comprimentoSorteado = np.random.random()*(comp_max-comp_min)+comp_min
    return comprimentoSorteado

def geraTree(grafo = None, inicio = 0, pontoInicial = None, anguloInicial = 45,  comp_min = 1, comp_max = 1, resolucao = 0.1, pGeometrias = [0.0, 0.05, 0.20, 0.05, 0.70], limiteXY = 150, funcaoDoCampo = projetaCampo):
    
    # Parametros iniciais do grafo: Ele precisa conter o primeiro ramo
    if grafo is None:
        grafo = nx.DiGraph()
    
    if pontoInicial is None:
        pontoInicial = np.array([0,0])

    comprimentoInicial =  comprimentoRamo(comp_min, comp_max)
    grafo.add_edges_from([(inicio,inicio+1)],tipo = "inicial", comprimento = comprimentoInicial, anguloLocal=0, anguloGlobal=anguloInicial)
    grafo.nodes[inicio]["pontoReal"] = pontoInicial
    grafo.nodes[inicio+1]["pontoReal"] = geraPonto(grafo, (inicio,inicio+1))

    ### T O P O L O G I A
    # Nomeia as geometrias e probabilidades
    Geometrias = ["Terminacao", "Campo de Repulsao", "Bifurcação em Y", "Bifurcação em T", "Continuacao"]

    # ESTRUTURA
    fila = [inicio+1] # Começa a fila a partir do 1
    visitado = []

    # PROBLEMA: 
    removidos = []

    while(fila):
        # Informações iniciais
        # Nó atual
        node = fila.pop(0)

        # Próximo nó
        proximo = list(grafo.nodes())[-1]+1
        # Edge
        edge = (node,proximo)
        
        # Aresta predessessora
        pred = list(grafo.predecessors(node))[0]
        predecessorEdge = (pred,node)

        # Escolhe qual vai ser a geometria escolhida
        index = int(np.random.choice(range(len(Geometrias)), size=1, p=pGeometrias))

        # Continuação
        if Geometrias[index] == "Continuacao":
            tipoEdge = "Continuacao"
            # PROXIMO: adicionar o edge dentro das funções de geometria para que nn tenham conflitos em caso ocorra a terminação
            grafo.add_edges_from([(node,proximo)], tipo = tipoEdge, comprimento = comprimentoRamo(comp_min, comp_max))
            grafo, flag = retilineo(grafo, predecessorEdge, edge, funcaoDoCampo, resolucao = resolucao)
            
            # Fila
            if flag != "Interrupcao":
                nodes = [proximo]
            else:
                nodes = []
    
        # Terminação
        elif Geometrias[index] == "Terminacao":
            tipoEdge = "Terminacao"
            # PROXIMO: adicionar o edge dentro das funções de geometria para que nn tenham conflitos em caso ocorra a terminação
            grafo.add_edges_from([(node,proximo)], tipo = tipoEdge, comprimento = comprimentoRamo(comp_min, comp_max))
            grafo, flag = retilineo(grafo, predecessorEdge, edge, funcaoDoCampo, resolucao = resolucao)

            # Fila
            nodes = []

        # Bifurcação em Y
        elif Geometrias[index] == "Bifurcação em Y":
            tipoEdge = "bif_Y_Direita"
            edge_Direito = (node,proximo)
            # PROXIMO: adicionar o edge dentro das funções de geometria para que nn tenham conflitos em caso ocorra a terminação
            grafo.add_edges_from([edge_Direito], tipo = tipoEdge, comprimento = comprimentoRamo(comp_min, comp_max))
            grafo,flag = bifurcacaoY(grafo, predecessorEdge, edge, funcaoDoCampo, resolucao = resolucao)
            
            if flag != "Interrupcao":
                tipoEdge = "bif_Y_Esquerda"
                edge_Esquerdo = (node,proximo+1)
                grafo.add_edges_from([edge_Esquerdo], tipo = tipoEdge, comprimento = comprimentoRamo(comp_min, comp_max))
                grafo, flag = bifurcacaoY(grafo, predecessorEdge, edge_Esquerdo, funcaoDoCampo, resolucao = resolucao)

           # Fila
            if flag != "Interrupcao":
                nodes = [proximo, proximo+1]
            else:
                nodes = []

        # Bifurcação em T
        elif Geometrias[index] == "Bifurcação em T":
            # PROXIMO: Criar função pra Bif_T
            tipoEdge = "bif_T"
            # PROXIMO: adicionar o edge dentro das funções de geometria para que nn tenham conflitos em caso ocorra a terminação
            grafo.add_edges_from([(node,proximo)], tipo = tipoEdge, comprimento = comprimentoRamo(comp_min, comp_max))
            intervalo = [85,95] # NECESSÁRIA PESQUISA
            thetaLocal = np.random.randint(*intervalo)
            grafo.edges[edge]["anguloLocal"] = thetaLocal
            grafo.edges[edge]["anguloGlobal"] = grafo.edges[predecessorEdge]["anguloGlobal"] + grafo.edges[edge]["anguloLocal"]               
            grafo.nodes[edge[1]]["pontoReal"] = geraPonto(grafo, edge)

            # Fila
            nodes = [proximo]

        # Campo de Repulsão
        elif Geometrias[index] == "Campo de Repulsao":
            tipoEdge = "campoRepulsao"
            # PROXIMO: adicionar o edge dentro das funções de geometria para que nn tenham conflitos em caso ocorra a terminação
            grafo.add_edges_from([(node,proximo)], tipo = tipoEdge, comprimento = comprimentoRamo(comp_min, comp_max))
            grafo, flag = campoDeRepulsão(grafo, predecessorEdge, edge, funcaoDoCampo, resolucao)
            
            # Fila
            if flag != "Interrupcao":
                nodes = [proximo]
            else:
                nodes = []

        # Adiciona nodes na fila
        for nodeFila in nodes:
            edge = (node,nodeFila)
            pontoAtual = grafo.nodes[nodeFila]["pontoReal"]
            x = pontoAtual[0]/resolucao
            y = pontoAtual[1]/resolucao
            flag = True
            if x <-limiteXY or x >limiteXY:
                removidos.append(nodeFila)
                tipo = grafo.edges[edge]["tipo"]
                #print(f"Edge removido: {edge}\tTipo: {tipo}")
                #print(f"Informações node {edge[0]}: ",grafo.nodes[edge[0]])
                #print(f"Informações node {edge[1]}: ",grafo.nodes[edge[1]])
                grafo.remove_edge(*edge)
                grafo.remove_node(edge[1]) # Remove só o último node pq o primeiro já está vinculado a um edge predecessor que não atingiu o limite
                flag = False
                #print(nodeFila,"-",pontoAtual)

            elif y <-limiteXY or y >limiteXY:
                removidos.append(nodeFila)
                tipo = grafo.edges[edge]["tipo"]
                #print(f"Edge removido: {edge}\tTipo: {tipo}")
                grafo.remove_edge(*edge)
                grafo.remove_node(edge[1]) # Remove só o último node pq o primeiro já está vinculado a um edge predecessor que não atingiu o limite
                flag = False
                #print(nodeFila,"-",pontoAtual)

            if flag == True:
                fila.append(nodeFila) # Adiciona os próximos na fila
          
        # Adiciona o node nos visitados
        visitado.append(node)
    #print(f"Removidos: {removidos}")
    #print(f"Nodes: {list(grafo.nodes())}")

    return grafo, removidos

def convertePixelParaPonto(grafo, pontoInicial: list, resolucao = 0.1):
    """
    INPUT
    -----
    grafo : dict, networkx.DiGraph() que tem os nós atuais do grafo e a característica "pontoReal" que são as coordenadas dos nós em valores decimais, ou seja, eixo x,y dos pontos; 
    
    pontoInicial : list(x,y), contém as cordenadas decimais do ponto que se espera projetar em pixel 
    
    resolucao : float, A resolução de uma imagem é o número de pontos por pixel em uma imagem


    OUTPUT
    -----
    pontoReal : tuple, contendo valores de x e y do ponto em unidades reais (eixo cartesiano)
    """
    xmin, ymin, semUso_dicPontos = cordenadasMinimosMaximos(grafo)
    
    x = pontoInicial[0]*resolucao
    y = pontoInicial[1]*resolucao
    x_new = x + xmin
    y_new = y + ymin
        
    x_final = x_new
    y_final = y_new

    return [x_final,y_final]

def continuaTree(grafo:dict, pGeometrias = [0.0, 0.05, 0.20, 0.05, 0.70], margem = 20, funcaoDoCampo = projetaCampoContinuacao, resolucao = 0.1, limiteXY = 150):
    # Parametros iniciais do segundo grafo:

    ## Escolhendo os valores de maximos possíveis para iniciar um novo ramo
    vasos, campo = funcaoDoCampo(grafo, margem = margem, resolucao=resolucao, raio = 5)
    maximo = np.max(campo)
    pontosMaximos = np.where(campo >= maximo)
    pontosMaximos = list(zip(*pontosMaximos))

    ## Selecionando as intensidades desses valores maximos
    intensidadesMaximas = []
    for ponto in pontosMaximos:
        intensidadesMaximas.append(campo[ponto])
    intensidadesMaximas = np.array(intensidadesMaximas)

    ## Com base na probabilidade exolhendo um index
    indexMaximo = retornaIndex(intensidades = intensidadesMaximas)
    if indexMaximo != "Interrupcao":
        ## Atribuindo valor da osição escolhido e convertendo pra cordenadas XY
        pontoPixel = pontosMaximos[indexMaximo]
        pontoPixel = pontoPixel[::-1] #Ponto em XY

        # Nó inicial do ramo
        pontoInicial = convertePixelParaPonto(grafo,pontoInicial=pontoPixel,resolucao=resolucao)
        inicio = list(grafo.nodes())[-1]+1

        ## Angulo do ramo novo
        angulo = np.random.randint(-180,180)
        grafo, rem = geraTree(grafo = grafo, inicio = inicio, pontoInicial = pontoInicial, anguloInicial = angulo, resolucao = 0.1, pGeometrias = pGeometrias, limiteXY = limiteXY, funcaoDoCampo = projetaCampo)
    else:
        print("Interrupção!!!!!")
    return vasos, campo





















