import sys
import json
import heapq # Essencial para a fila de prioridade no A*
import os
import folium # Usado para criar os mapas interativos
import tempfile # Ajuda a criar arquivos temporários, como o HTML do mapa
import math # Para cálculos de distância (usado na heurística)
import traceback # Útil para imprimir mensagens de erro detalhadas
import webbrowser # Para abrir o mapa no navegador padrão
from typing import List, Tuple, Dict, Optional, Any, Set, Union # Para type hints, deixando o código mais claro

# --- Configuração da Renderização Alternativa ---
# Força o Qt a usar renderização por software, mais lenta mas muito mais compatível
os.environ['QT_OPENGL'] = 'software'
print(f"Configurando renderização alternativa: QT_OPENGL=software")

# --- Importações PyQt6: Tudo que precisamos para a interface gráfica ---
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QComboBox, QPushButton, QTextEdit, QMessageBox,
                             QSizePolicy, QSplitter, QScrollArea) # QCompleter não é mais necessário
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon, QCursor # Para estilos e ícones
from PyQt6.QtCore import Qt, QCoreApplication, QUrl, QTimer # Funcionalidades centrais, URLs e timers

# --- Configurando o Visualizador de Mapa (QWebEngineView) ---
# Mesmo que não usemos mais para exibir, a importação pode ser útil para referência ou futuras mudanças.
# Vamos manter a verificação, mas o código principal usará webbrowser.
WEB_ENGINE_AVAILABLE: bool = False # Vamos assumir que não está disponível inicialmente
QWebEngineView: Optional[type] = None # Espaço reservado para a classe do widget de visualização do mapa
QWebEngineProfile: Optional[type] = None # Espaço reservado para configurações do perfil do navegador
QWebEnginePage: Optional[type] = None # Espaço reservado para o objeto da página web

try:
    # Tentamos importar as classes necessárias dos módulos WebEngine do PyQt6
    from PyQt6.QtWebEngineWidgets import QWebEngineView as ActualQWebEngineView # type: ignore
    from PyQt6.QtWebEngineCore import QWebEngineProfile as ActualQWebEngineProfile, QWebEnginePage as ActualQWebEnginePage # type: ignore

    # Se as importações funcionaram, atribuímos as classes reais aos nossos espaços reservados
    QWebEngineView = ActualQWebEngineView
    QWebEngineProfile = ActualQWebEngineProfile
    QWebEnginePage = ActualQWebEnginePage

    # Verificação dupla para garantir que QWebEngineView foi atribuído com sucesso
    if QWebEngineView is not None:
        WEB_ENGINE_AVAILABLE = True # Eba! Podemos usar o web engine para os mapas.
        print("Debug: PyQt6.QtWebEngineWidgets importado com sucesso (mas não será usado para exibição principal).")
    else:
        # Este caso é improvável se a importação teve sucesso, mas é bom tratar
        WEB_ENGINE_AVAILABLE = False
        print("Aviso: QWebEngineView é None mesmo após importação. Funcionalidade de mapa interno desabilitada.")

except ImportError:
    # Isso acontece se o usuário não tiver o PyQt6-WebEngine instalado
    print("*"*20)
    print("Aviso: PyQt6.QtWebEngineWidgets não encontrado.")
    print("Por favor, instale 'PyQt6-WebEngine' usando: pip install PyQt6-WebEngine")
    print("A funcionalidade de mapa interno será desabilitada (usaremos o navegador externo).")
    print("*"*20)
    # Garantimos que nossos espaços reservados sejam None se a importação falhar
    QWebEngineView = None
    QWebEngineProfile = None
    QWebEnginePage = None
    WEB_ENGINE_AVAILABLE = False
except Exception as e:
    # Captura quaisquer outros erros inesperados durante a importação ou configuração
    print(f"Aviso: Ocorreu um erro inesperado ao importar/inicializar PyQtWebEngine: {e}")
    traceback.print_exc() # Imprime os detalhes completos do erro para depuração
    # Garante que a funcionalidade do mapa interno está desabilitada
    QWebEngineView = None
    QWebEngineProfile = None
    QWebEnginePage = None
    WEB_ENGINE_AVAILABLE = False


# --- Implementação do Algoritmo A* ---
class Node:
    """
    Representa uma localização (cidade) em nossa busca. Pense nisso como um passo no caminho potencial.
    Ele armazena o nome da cidade, o passo anterior (nó pai) que levou até aqui,
    e os custos associados para chegar a esta cidade.
    """
    def __init__(self, parent: Optional['Node'], position: str):
        self.parent: Optional['Node'] = parent # O nó de onde viemos para chegar aqui
        self.position: str = position # O nome da cidade que este nó representa
        self.g: float = 0.0 # Custo 'g': O custo real acumulado da cidade inicial até esta
        self.h: float = 0.0 # Custo 'h': O custo heurístico estimado desta cidade até o destino final
        self.f: float = 0.0 # Custo 'f': O custo total estimado (g + h). A* prioriza nós com menor custo 'f'.

    def __eq__(self, other: object) -> bool:
        # Define como comparar se dois nós são "iguais" (baseado na posição/cidade)
        if not isinstance(other, Node):
            return NotImplemented # Não sabe comparar com outros tipos
        return self.position == other.position

    def __lt__(self, other: 'Node') -> bool:
        # Define como comparar qual nó é "menor" (mais promissor) para a fila de prioridade.
        # Primeiro, compara pelo custo total 'f'. Menor 'f' é melhor.
        if self.f != other.f:
            return self.f < other.f
        # Se os custos 'f' são iguais, desempata usando a heurística 'h'. Menor 'h' é melhor (mais perto do destino).
        return self.h < other.h
        # Alternativa de desempate (raramente usada): maior 'g', favorece caminhos já mais longos.
        # return self.g > other.g

    def __hash__(self) -> int:
        # Permite que nós sejam usados em conjuntos (sets) ou como chaves de dicionário, baseado na posição.
        return hash(self.position)

    def __repr__(self) -> str:
        # Como o nó deve ser representado em texto (útil para debug e logs).
        # Formata os números para facilitar a leitura.
        return f"{self.position:<15} (f={self.f:<6.1f}, g={self.g:<6.1f}, h={self.h:<6.1f})"

def calculate_heuristic(pos1_coords: Optional[List[float]], pos2_coords: Optional[List[float]]) -> float:
    """
    Calcula a distância Euclidiana (linha reta) entre duas coordenadas geográficas.
    Isso serve como nossa estimativa 'h' no A*.
    Retorna 0.0 se alguma coordenada for inválida ou ausente.
    """
    # Verifica se temos duas coordenadas válidas (listas com 2 números)
    if pos1_coords and pos2_coords and len(pos1_coords) == 2 and len(pos2_coords) == 2:
        try:
            # Converte as coordenadas para números (float)
            lat1, lon1 = float(pos1_coords[0]), float(pos1_coords[1])
            lat2, lon2 = float(pos2_coords[0]), float(pos2_coords[1])
            # Aplica a fórmula da distância Euclidiana
            dist = math.sqrt(((lat1 - lat2) ** 2) + ((lon1 - lon2) ** 2))
            # Fator de escala opcional: A distância em graus é pequena. Multiplicar
            # pode dar mais peso à heurística. Ajuste conforme necessário.
            scale_factor = 100.0 # Exemplo: Amplifica a heurística
            return dist * scale_factor
        except (TypeError, ValueError) as e:
            # Se as coordenadas não forem números válidos
            print(f"Aviso: Erro ao calcular heurística entre {pos1_coords} e {pos2_coords}: {e}")
            return 0.0 # Retorna 0 em caso de erro
    # Se as coordenadas não foram fornecidas ou têm formato inválido
    # print(f"Aviso: Coordenadas inválidas para cálculo de heurística: {pos1_coords}, {pos2_coords}")
    return 0.0

def astar(graph: Dict[str, Dict[str, float]],
          heuristics: Dict[str, float],
          coordinates: Dict[str, List[float]],
          start: str,
          end: str,
          use_dynamic_heuristic: bool = True) -> Tuple[Optional[List[str]], float, List[str]]:
    """
    Executa o algoritmo A* para encontrar o caminho mais curto entre 'start' e 'end'.

    Retorna:
    - Uma lista com as cidades do caminho (se encontrado), ou None.
    - O custo total do caminho (se encontrado), ou infinito.
    - Um log detalhado dos passos da busca (quais nós foram expandidos).
    """
    expanded_nodes_log: List[str] = [] # Onde guardaremos os detalhes da busca

    # --- Verificações Iniciais ---
    # A cidade de início *precisa* existir nas conexões do grafo.
    if start not in graph:
        print(f"Erro A*: Cidade de início '{start}' não encontrada nas chaves de 'connections'.")
        expanded_nodes_log.append(f"ERRO: Cidade de início '{start}' não encontrada no grafo.")
        return None, float('inf'), expanded_nodes_log
    # O destino não *precisa* ter conexões de saída, pode ser um ponto final.
    # if end not in graph: # Comentado pois o destino pode não ter saídas
    #     print(f"Erro A*: Cidade de destino '{end}' não encontrada nas chaves de 'connections'.")
    #     return None, float('inf'), expanded_nodes_log

    # Verifica se temos coordenadas, especialmente se vamos calcular a heurística dinamicamente.
    if start not in coordinates or end not in coordinates:
         if use_dynamic_heuristic:
             print(f"Aviso A*: Coordenadas para '{start}' ou '{end}' não encontradas. Heurística dinâmica será 0.")
         # Se não for usar heurística dinâmica, verifica se temos a heurística pré-calculada.
         elif start not in heuristics or end not in heuristics:
              print(f"Aviso A*: Heurísticas pré-calculadas para '{start}' ou '{end}' não encontradas.")

    # Caso simples: já estamos no destino?
    if start == end:
        expanded_nodes_log.append(f"- Nó inicial e final são os mesmos: {start}")
        return [start], 0.0, expanded_nodes_log

    # --- Preparação para a Busca ---
    # Cria os nós inicial e final. O final é mais para referência.
    start_node = Node(None, start) # Nó inicial não tem pai
    end_node = Node(None, end)   # Nó final (só precisamos da posição)

    # Calcula a heurística (estimativa) do nó inicial até o destino.
    start_coords = coordinates.get(start)
    end_coords = coordinates.get(end)
    if use_dynamic_heuristic:
        # Calcula na hora usando as coordenadas (distância em linha reta).
        start_node.h = calculate_heuristic(start_coords, end_coords)
    else:
        # Usa o valor de heurística que veio do arquivo JSON.
        start_node.h = heuristics.get(start, 0.0) # Usa 0.0 se não encontrar

    # Custos iniciais: g=0 (custo zero para chegar ao início), f = g + h
    start_node.g = 0.0
    start_node.f = start_node.g + start_node.h

    # --- Estruturas de Dados do A* ---
    # Lista Aberta (Open List): Nós a serem explorados, organizada como uma fila de prioridade (heap).
    # O nó com menor 'f' sempre sai primeiro.
    open_list: List[Node] = []
    # Dicionário Aberto (Open Dict): Ajuda a acessar rapidamente um nó na lista aberta pela sua posição
    # e a verificar/atualizar custos se encontrarmos um caminho melhor para um nó já na lista.
    open_dict: Dict[str, Node] = {}
    # Conjunto Fechado (Closed Set): Guarda as posições (nomes das cidades) que já foram totalmente exploradas.
    # Não precisamos revisitar nós no conjunto fechado.
    closed_set: Set[str] = set()

    # --- Início da Busca ---
    # Adiciona o nó inicial à lista aberta.
    heapq.heappush(open_list, start_node)
    open_dict[start_node.position] = start_node
    expanded_nodes_log.append(f"- Iniciando busca em: {start_node.position} (f={start_node.f:.1f})")

    # --- Loop Principal do A* ---
    # Continua enquanto houver nós promissores na lista aberta.
    while open_list:
        # 1. Pega o nó mais promissor (menor 'f') da lista aberta.
        try:
            current_node = heapq.heappop(open_list)
            # Verificação de segurança: Este nó ainda é o melhor caminho conhecido para esta posição?
            # Pode acontecer de termos adicionado um caminho melhor para esta mesma cidade
            # depois que este nó já estava na fila. Se sim, ignoramos este nó "obsoleto".
            if current_node.position not in open_dict or open_dict[current_node.position].f < current_node.f:
                continue # Pula para o próximo nó da lista aberta

        except IndexError:
            # Se a lista aberta ficar vazia, significa que exploramos tudo e não achamos o destino.
            expanded_nodes_log.append("- Lista aberta vazia, destino não alcançado.")
            break # Fim da busca (sem sucesso)

        # 2. Log: Registra qual nó estamos expandindo agora.
        parent_info = f" (Veio de: {current_node.parent.position})" if current_node.parent else " (Nó inicial)"
        log_entry = f"- Expandindo: {current_node.position:<15} | f={current_node.f:<6.1f} | g={current_node.g:<6.1f} | h={current_node.h:<6.1f}{parent_info}"
        expanded_nodes_log.append(log_entry)

        # 3. Chegou ao Destino?
        if current_node.position == end_node.position:
            # Sucesso! Reconstrói o caminho voltando pelos pais.
            path: List[str] = []
            total_cost: float = current_node.g # O custo 'g' do nó final é o custo total real
            temp: Optional[Node] = current_node
            while temp is not None:
                path.append(temp.position)
                temp = temp.parent # Vai para o nó anterior
            expanded_nodes_log.append(f"- Destino '{end}' encontrado! Custo: {total_cost:.1f}")
            return path[::-1], total_cost, expanded_nodes_log # Retorna o caminho na ordem correta

        # 4. Mover para o Conjunto Fechado: Marca o nó atual como explorado.
        closed_set.add(current_node.position)
        # Remove do dicionário aberto (se ainda estava lá com este custo exato)
        if current_node.position in open_dict and open_dict[current_node.position] == current_node:
             del open_dict[current_node.position]

        # 5. Explorar Vizinhos: Olha as cidades conectadas ao nó atual.
        # Verifica se a cidade atual tem vizinhos definidos no grafo.
        if current_node.position not in graph:
            expanded_nodes_log.append(f"  - Aviso: Nó '{current_node.position}' não possui conexões de saída.")
            continue # Pula para o próximo nó na lista aberta

        neighbors_evaluated = 0 # Contador de vizinhos válidos processados
        for neighbor, step_cost in graph[current_node.position].items():
            # 'neighbor' é o nome da cidade vizinha, 'step_cost' é o custo para ir do nó atual até ela.

            # a. Ignora vizinho se já foi explorado (está no conjunto fechado).
            if neighbor in closed_set:
                # expanded_nodes_log.append(f"  - Vizinho '{neighbor}' já está no conjunto fechado. Ignorando.")
                continue

            # b. Calcula o custo 'g' para chegar ao vizinho *passando pelo nó atual*.
            tentative_g = current_node.g + step_cost

            # c. Verifica se já conhecemos um caminho para este vizinho (se ele está na lista aberta).
            existing_node_in_open = open_dict.get(neighbor)

            update_or_add = True # Flag: Devemos adicionar/atualizar este vizinho na lista aberta?
            log_neighbor_status = "" # Mensagem para o log sobre o que aconteceu com o vizinho

            if existing_node_in_open:
                # Se o vizinho já está na lista aberta:
                if existing_node_in_open.g <= tentative_g:
                    # O caminho que já conhecíamos para ele é melhor ou igual a este novo caminho. Ignora.
                    # log_neighbor_status = f"(Caminho via {existing_node_in_open.parent.position if existing_node_in_open.parent else '?'} com g={existing_node_in_open.g:.1f} é melhor/igual. Ignorando este)"
                    update_or_add = False
                else:
                    # Este novo caminho (passando pelo nó atual) é MELHOR! Vamos atualizar.
                    log_neighbor_status = f"(Atualizando caminho anterior com g={existing_node_in_open.g:.1f})"
                    # Não precisamos remover o nó antigo explicitamente da heap (fila).
                    # A verificação no início do loop (passo 1) vai ignorar o nó antigo (com 'f' maior)
                    # quando ele eventualmente chegar ao topo da heap. Apenas adicionamos o novo melhor nó.
            else:
                # Se o vizinho não estava na lista aberta, é a primeira vez que o alcançamos.
                log_neighbor_status = "(Adicionando à lista aberta)"

            # d. Adiciona ou Atualiza o Vizinho na Lista Aberta:
            if update_or_add:
                neighbors_evaluated += 1
                # Cria um novo nó para o vizinho.
                child_node = Node(current_node, neighbor) # O pai é o nó atual
                child_node.g = tentative_g # Custo real acumulado até ele

                # Calcula a heurística 'h' do vizinho até o destino final.
                neighbor_coords = coordinates.get(neighbor)
                if use_dynamic_heuristic:
                    child_node.h = calculate_heuristic(neighbor_coords, end_coords)
                else:
                    child_node.h = heuristics.get(neighbor, 0.0)

                # Calcula o custo total 'f'.
                child_node.f = child_node.g + child_node.h

                # Adiciona o nó vizinho (ou sua versão atualizada) à fila de prioridade e ao dicionário.
                heapq.heappush(open_list, child_node)
                open_dict[neighbor] = child_node # O dicionário agora aponta para este nó (o melhor caminho conhecido até agora)

                # Log opcional (pode ficar muito verboso):
                # expanded_nodes_log.append(f"  -> Avaliado: {neighbor:<15} | f={child_node.f:<6.1f} | g={child_node.g:<6.1f} | h={child_node.h:<6.1f} {log_neighbor_status}")

        # Log opcional se um nó não teve vizinhos válidos (todos já estavam no conjunto fechado)
        if neighbors_evaluated == 0 and current_node.position != end:
             # expanded_nodes_log.append(f"  - Nenhum vizinho válido (não fechado) encontrado para {current_node.position}.")
             pass # Não faz nada, só continua a busca

    # --- Fim da Busca (Sem Sucesso) ---
    # Se o loop terminou porque a lista aberta ficou vazia.
    print(f"A* completou a busca mas não encontrou caminho de {start} para {end}.")
    expanded_nodes_log.append(f"- Fim da busca: Destino '{end}' não encontrado a partir de '{start}'.")
    return None, float('inf'), expanded_nodes_log


# --- Interface Gráfica com PyQt6 ---
class AStarApp(QWidget):
    """
    A classe principal que monta e controla a janela da nossa aplicação A*.
    """
    def __init__(self, graph_data: Dict[str, Any]):
        super().__init__() # Inicializa a classe base QWidget
        print("Debug: Iniciando AStarApp.__init__")
        # --- Atributos da Classe ---
        # Onde guardaremos os dados do grafo, heurísticas e coordenadas
        self.graph: Dict[str, Dict[str, float]] = {}
        self.heuristics: Dict[str, float] = {}
        self.coordinates: Dict[str, List[float]] = {}
        self.all_cities: List[str] = [] # Lista com todas as cidades disponíveis
        self.map_file_path: str = "" # Caminho para o arquivo HTML temporário do mapa

        # Referências para os widgets da interface (menus, caixas de texto, etc.)
        self.originCombo: Optional[QComboBox] = None
        self.destCombo: Optional[QComboBox] = None
        self.stopCombos: List[QComboBox] = [] # Lista para os menus de parada
        self.resultText: Optional[QTextEdit] = None # Caixa de texto para os resultados
        self.stepsText: Optional[QTextEdit] = None # Caixa de texto para os detalhes do A*
        # MODIFICAÇÃO: mapView será sempre um QLabel agora
        self.mapView: Optional[QLabel] = None # Onde o aviso sobre o mapa será exibido

        # --- Carregamento e Validação dos Dados ---
        print("Debug: Chamando _validate_and_load_graph_data")
        # Tenta carregar e validar os dados recebidos (do arquivo JSON)
        if not self._validate_and_load_graph_data(graph_data):
             # Se a validação falhar em algo crítico, impede a criação da janela.
             raise ValueError("Falha ao validar ou carregar dados essenciais do grafo.")
        print("Debug: _validate_and_load_graph_data concluído")

        # --- Configuração do Mapa Temporário ---
        # Define um nome único para o arquivo HTML do mapa (usando o ID do processo)
        pid = os.getpid()
        self.map_file_path = os.path.join(tempfile.gettempdir(), f"astar_map_{pid}.html")
        print(f"Debug: Caminho do mapa temporário definido: {self.map_file_path}")

        # --- Montagem da Interface ---
        print("Debug: Chamando initUI")
        self.initUI() # Chama a função que cria todos os botões, menus, etc.
        print("Debug: initUI concluído")

        # O mapa inicial será gerado e aberto no navegador *depois* que a janela for exibida.
        # self.create_initial_map() # Não chamamos aqui
        print("Debug: AStarApp.__init__ concluído")


    def _validate_and_load_graph_data(self, graph_data: Dict[str, Any]) -> bool:
        """
        Verifica se os dados carregados do JSON estão no formato esperado e os armazena.
        Retorna True se os dados essenciais são válidos, False caso contrário.
        """
        # Verifica se temos as chaves principais ('connections', 'heuristics', 'coordinates')
        required_keys = ['connections', 'heuristics', 'coordinates']
        if not graph_data or any(key not in graph_data for key in required_keys):
            print(f"Erro Crítico: Dados do grafo inválidos. Chaves obrigatórias ausentes: {required_keys}")
            return False # Erro fatal

        # Verifica se os valores dessas chaves são dicionários
        if not isinstance(graph_data.get('connections'), dict) or \
           not isinstance(graph_data.get('heuristics'), dict) or \
           not isinstance(graph_data.get('coordinates'), dict):
             print("Erro Crítico: Formato inválido para 'connections', 'heuristics' ou 'coordinates'. Devem ser dicionários.")
             return False # Erro fatal

        # Armazena os dados principais
        self.graph = graph_data['connections']
        self.heuristics = graph_data['heuristics']
        self.coordinates = graph_data['coordinates']

        # --- Validação Detalhada e Limpeza ---
        # Pega todas as cidades mencionadas nas conexões (tanto como origem quanto destino)
        cities_in_connections_keys: Set[str] = set(self.graph.keys())
        cities_in_connections_values: Set[str] = set()
        valid_graph: Dict[str, Dict[str, float]] = {} # Novo dicionário para guardar apenas conexões válidas

        # Itera sobre as conexões para validar vizinhos e custos
        for city, neighbors in self.graph.items():
            if isinstance(neighbors, dict): # O valor deve ser um dicionário de vizinhos
                valid_neighbors: Dict[str, float] = {}
                cities_in_connections_values.update(neighbors.keys()) # Adiciona vizinhos ao conjunto
                # Valida cada vizinho e custo
                for neighbor, cost in neighbors.items():
                    # O custo deve ser um número não negativo
                    if isinstance(cost, (int, float)) and cost >= 0:
                         valid_neighbors[neighbor] = float(cost) # Armazena como float
                    else:
                        # Se o custo for inválido, ignora essa conexão específica
                        print(f"Aviso nos Dados: Custo inválido ({cost}) de {city} para {neighbor}. Conexão ignorada.")
                # Só adiciona a cidade ao grafo válido se ela tiver pelo menos um vizinho válido
                if valid_neighbors:
                     valid_graph[city] = valid_neighbors
            else:
                # Se o valor para uma cidade não for um dicionário
                print(f"Aviso nos Dados: Valor inválido em 'connections' para '{city}': esperado dicionário de vizinhos, encontrado {type(neighbors)}. '{city}' ignorada.")

        # Atualiza o grafo da instância para conter apenas as conexões válidas
        self.graph = valid_graph
        # Recalcula os conjuntos de cidades válidas nas conexões
        cities_in_connections_keys = set(self.graph.keys())
        cities_in_connections_values = set(n for neighbors in self.graph.values() for n in neighbors)

        # Pega as cidades mencionadas nas heurísticas e coordenadas
        cities_in_heuristics: Set[str] = set(self.heuristics.keys())
        cities_in_coords: Set[str] = set(self.coordinates.keys())

        # Cria a lista final de TODAS as cidades únicas encontradas em qualquer lugar, ordenadas.
        self.all_cities = sorted(list(
            cities_in_connections_keys.union(cities_in_connections_values)
                                   .union(cities_in_heuristics)
                                   .union(cities_in_coords)
        ))

        # Se, após toda a validação, não sobrar nenhuma cidade, é um erro crítico.
        if not self.all_cities:
            print("Erro Crítico: Nenhuma cidade válida encontrada após validação dos dados.")
            return False

        # --- Validação Adicional e Limpeza de Heurísticas/Coordenadas ---
        valid_heuristics: Dict[str, float] = {} # Para guardar heurísticas válidas
        valid_coordinates: Dict[str, List[float]] = {} # Para guardar coordenadas válidas
        missing_coords: Set[str] = set() # Cidades do grafo sem coordenadas válidas
        missing_heuristics: Set[str] = set() # Cidades sem heurísticas válidas

        # Itera sobre todas as cidades encontradas para validar seus dados individuais
        for city in self.all_cities:
            # Valida Coordenadas: Devem ser uma lista/tupla de 2 números.
            coord = self.coordinates.get(city)
            if coord and isinstance(coord, list) and len(coord) == 2 and all(isinstance(c, (int, float)) for c in coord):
                valid_coordinates[city] = [float(coord[0]), float(coord[1])] # Armazena como floats
            else:
                # Se as coordenadas são inválidas ou ausentes:
                # Só registra como 'faltando' se a cidade realmente é usada nas conexões.
                if city in cities_in_connections_keys or city in cities_in_connections_values:
                     missing_coords.add(city)
                # Se a chave existia mas o valor era inválido, imprime um aviso.
                if city in self.coordinates:
                     print(f"Aviso nos Dados: Coordenadas inválidas para '{city}': {coord}. Ignoradas.")

            # Valida Heurísticas: Devem ser um número não negativo.
            h_val = self.heuristics.get(city)
            if isinstance(h_val, (int, float)) and h_val >= 0:
                valid_heuristics[city] = float(h_val) # Armazena como float
            else:
                 # Se a heurística é inválida ou ausente:
                 missing_heuristics.add(city) # Adiciona ao conjunto de faltantes
                 # Se a chave existia mas o valor era inválido, imprime um aviso.
                 if city in self.heuristics:
                     print(f"Aviso nos Dados: Heurística inválida para '{city}': {h_val}. Ignorada.")

        # Atualiza os dicionários da instância com os dados validados/limpos
        self.coordinates = valid_coordinates
        self.heuristics = valid_heuristics
        # Nota: self.all_cities ainda contém todas as cidades originalmente encontradas,
        # mesmo que algumas não tenham coordenadas ou heurísticas válidas.

        # Imprime avisos consolidados sobre dados faltantes (mas não fatais)
        if missing_coords:
            print(f"Aviso: Faltando coordenadas válidas para cidades usadas no grafo: {', '.join(sorted(list(missing_coords)))}. Heurística dinâmica pode ser 0.")
        # O aviso sobre heurísticas faltantes é menos crítico se usarmos a heurística dinâmica.
        # if missing_heuristics:
        #      print(f"Aviso: Faltando heurísticas pré-calculadas válidas para: {', '.join(sorted(list(missing_heuristics)))}.")

        # Verificação final de consistência: Uma cidade usada no grafo precisa ter
        # pelo menos coordenadas (para heurística dinâmica) OU uma heurística pré-calculada válida.
        cities_in_valid_graph = set(self.graph.keys()).union(set(n for neighbors in self.graph.values() for n in neighbors))
        for city in cities_in_valid_graph:
             if city not in self.coordinates and city not in self.heuristics:
                 # Isso pode ser um problema sério dependendo da configuração.
                 print(f"Aviso Crítico: Cidade '{city}' está no grafo mas não possui coordenada (para heurística dinâmica) nem heurística pré-calculada válida.")
                 # Poderíamos retornar False aqui se isso for considerado um erro que impede a execução.

        return True # Se chegou até aqui, os dados essenciais estão OK.

    def initUI(self) -> None:
        """Monta a interface gráfica da aplicação (janela, botões, menus, etc.)."""
        self.setWindowTitle('A* Busca de Caminho com Paradas e Mapa') # Título da janela
        self.setGeometry(50, 50, 1200, 800) # Posição inicial (X, Y) e tamanho (Largura, Altura)

        # Aplica um estilo visual usando CSS (folha de estilos)
        self.setStyleSheet("""
            QWidget { /* Estilo geral para todos os widgets */
                background-color: #f0f4f8; /* Fundo azul claro acinzentado */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Fonte padrão */
                font-size: 10pt; /* Tamanho da fonte padrão */
            }
            QLabel { /* Estilo para os rótulos (textos descritivos) */
                font-size: 10pt; color: #1c2833; /* Cor cinza escuro */
                padding-bottom: 2px; /* Pequeno espaço abaixo */
            }
            QComboBox { /* Estilo para os menus dropdown (caixas de combinação) */
                font-size: 10pt; padding: 7px 9px; /* Espaçamento interno */
                border: 1px solid #aeb6bf; /* Borda cinza */
                border-radius: 5px; /* Cantos arredondados */
                background-color: white; /* Fundo branco */
                min-height: 26px; /* Altura mínima */
            }
            QComboBox:disabled { /* Estilo quando o menu está desabilitado */
                background-color: #e5e8e8; /* Fundo cinza claro */
                color: #99a3a4; /* Texto cinza */
            }
            QComboBox::drop-down { /* Estilo da setinha do dropdown */
                subcontrol-origin: padding; subcontrol-position: top right; width: 20px;
                border-left-width: 1px; border-left-color: #aeb6bf; border-left-style: solid;
                border-top-right-radius: 5px; border-bottom-right-radius: 5px;
            }
            QComboBox QAbstractItemView { /* Estilo da lista que aparece ao clicar no dropdown */
                border: 1px solid #aeb6bf; background-color: white;
                selection-background-color: #5dade2; /* Cor de fundo do item selecionado (azul) */
                selection-color: white; /* Cor do texto do item selecionado (branco) */
                color: #1c2833; padding: 5px; outline: 0px; /* Remove a linha pontilhada de foco */
            }
            QPushButton { /* Estilo para os botões */
                font-size: 10pt; font-weight: bold; color: white; /* Texto branco e negrito */
                background-color: #3498db; /* Fundo azul */
                padding: 10px 20px; /* Espaçamento interno */
                border: none; border-radius: 5px; /* Sem borda, cantos arredondados */
                min-width: 140px; /* Largura mínima */
                outline: 0px; /* Remove a linha pontilhada de foco */
            }
            QPushButton:hover { background-color: #2980b9; } /* Cor do botão ao passar o mouse */
            QPushButton:pressed { background-color: #1f618d; } /* Cor do botão ao clicar */
            QTextEdit { /* Estilo para as caixas de texto (resultados e detalhes) */
                font-size: 9.5pt; border: 1px solid #aeb6bf; border-radius: 5px;
                background-color: #fdfefe; /* Fundo quase branco */
                color: #2c3e50; /* Cor do texto cinza azulado */
                padding: 8px; /* Espaçamento interno */
            }
            QSplitter::handle { background-color: #bdc3c7; } /* Cor da barrinha divisora */
            QSplitter::handle:horizontal { width: 6px; } /* Largura da barrinha horizontal */
            QSplitter::handle:vertical { height: 6px; } /* Altura da barrinha vertical */
            QScrollArea { border: none; } /* Remove borda de áreas de rolagem (se usadas) */
            #titleLabel { /* Estilo específico para o título principal (usando ID) */
                font-size: 16pt; font-weight: bold; color: #2980b9; /* Fonte maior, negrito, azul */
                padding-bottom: 15px; qproperty-alignment: 'AlignCenter'; /* Centralizado */
            }
            #resultLabel, #mapLabel, #stepsLabel { /* Estilo para os títulos das seções (Resultados, Mapa, Detalhes) */
                font-size: 11pt; font-weight: bold; color: #1c2833; /* Fonte um pouco maior, negrito */
                padding-top: 12px; padding-bottom: 6px; /* Espaçamento vertical */
            }
            /* Estilo específico para o QLabel do mapa */
            #mapPlaceholderLabel {
                background-color: #dde;
                border: 1px solid #aac;
                padding: 20px;
                color: #555;
                font-style: italic;
            }
        """)

        # --- Estrutura da Janela ---
        # Layout principal que organiza tudo verticalmente
        mainLayout = QVBoxLayout(self)
        # Divisor (Splitter) que separa a janela em duas partes: controle (esquerda) e resultados/mapa (direita)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Cria os painéis esquerdo e direito chamando funções auxiliares
        print("Debug: Criando painel de controle")
        controlPanelWidget = self._create_control_panel() # Painel com menus e botão
        print("Debug: Criando painel de resultados/mapa")
        resultsMapWidget = self._create_results_map_panel() # Painel com textos e mapa
        print("Debug: Painéis criados")

        # Adiciona os painéis ao divisor
        splitter.addWidget(controlPanelWidget)
        splitter.addWidget(resultsMapWidget)
        # Define como o espaço extra deve ser distribuído (1 parte para controle, 3 para resultados/mapa)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([350, 850]) # Sugestão de tamanhos iniciais

        # Adiciona o divisor ao layout principal da janela
        mainLayout.addWidget(splitter)
        mainLayout.setContentsMargins(15, 15, 15, 15) # Margens ao redor de todo o conteúdo da janela

        # Coloca o foco inicial no menu de seleção de origem
        if self.originCombo:
            self.originCombo.setFocus()

        # Configura o estado inicial dos menus (desabilita destino e paradas até a origem ser escolhida)
        print("Debug: Atualizando estado inicial dos combos")
        self._update_destination_options()
        self._update_stop_options()
        print("Debug: Estado inicial dos combos atualizado")


    def _create_control_panel(self) -> QWidget:
        """Cria o painel esquerdo da interface, com os menus e o botão de busca."""
        controlPanelWidget = QWidget() # O container do painel
        controlPanelLayout = QVBoxLayout(controlPanelWidget) # Organiza os itens verticalmente
        controlPanelWidget.setMinimumWidth(350) # Garante uma largura mínima
        controlPanelWidget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding) # Como ele se ajusta

        # Título do painel
        titleLabel = QLabel('Busca de Caminho A*')
        titleLabel.setObjectName("titleLabel") # ID para aplicar estilo CSS específico

        # Layout em grade para organizar os rótulos e menus (Origem, Destino, Paradas)
        gridLayout = QGridLayout()
        gridLayout.setVerticalSpacing(12) # Espaço vertical entre linhas
        gridLayout.setHorizontalSpacing(10) # Espaço horizontal entre colunas

        # --- Widget de Origem ---
        originLabel = QLabel('Origem:')
        self.originCombo = QComboBox() # Cria o menu dropdown
        # Preenche o menu com as cidades. Adiciona uma opção vazia no início.
        cities_list = [""] + (self.all_cities if self.all_cities else [])
        self.originCombo.addItems(cities_list)
        self.originCombo.setToolTip("Selecione a cidade inicial da rota") # Dica ao passar o mouse
        self.originCombo.setEditable(False) # Impede que o usuário digite no menu
        # Conecta a ação de mudar a seleção à função que atualiza o menu de destino
        self.originCombo.currentIndexChanged.connect(self._update_destination_options)

        # --- Widget de Destino ---
        destLabel = QLabel('Destino:')
        self.destCombo = QComboBox()
        self.destCombo.addItems([""]) # Começa vazio
        self.destCombo.setToolTip("Selecione a cidade final da rota (após escolher a origem)")
        self.destCombo.setEditable(False)
        self.destCombo.setEnabled(False) # Começa desabilitado (até escolher a origem)
        # Conecta a ação de mudar a seleção à função que atualiza os menus de parada
        self.destCombo.currentIndexChanged.connect(self._update_stop_options)

        # Adiciona os widgets de origem e destino na grade (linha, coluna)
        gridLayout.addWidget(originLabel, 0, 0); gridLayout.addWidget(self.originCombo, 0, 1)
        gridLayout.addWidget(destLabel, 1, 0); gridLayout.addWidget(self.destCombo, 1, 1)

        # --- Widgets de Parada ---
        self.stopCombos = [] # Lista para guardar os menus de parada
        for i in range(3): # Cria 3 menus de parada
            stopLabel = QLabel(f'Parada {i+1} (Opcional):')
            stopCombo = QComboBox()
            stopCombo.addItems([""]) # Começa vazio
            stopCombo.setToolTip(f"Selecione a parada opcional {i+1} (após origem e destino)")
            stopCombo.setEditable(False)
            stopCombo.setEnabled(False) # Começa desabilitado
            # Conecta a mudança de seleção para atualizar os próximos menus de parada (lógica em cascata)
            stopCombo.currentIndexChanged.connect(self._update_stop_options)
            # Adiciona na grade
            gridLayout.addWidget(stopLabel, i + 2, 0)
            gridLayout.addWidget(stopCombo, i + 2, 1)
            self.stopCombos.append(stopCombo) # Guarda a referência do menu

        # Faz a segunda coluna (dos menus) esticar para preencher o espaço horizontal
        gridLayout.setColumnStretch(1, 1)

        # --- Botão de Busca ---
        buttonLayout = QHBoxLayout() # Layout para centralizar o botão
        findButton = QPushButton('Encontrar Caminho e Abrir Mapa') # Texto do botão atualizado
        findButton.setToolTip("Calcular a rota incluindo paradas e abrir o mapa no navegador") # Dica atualizada
        # Conecta o clique do botão à função principal de cálculo e exibição
        findButton.clicked.connect(self.find_path_and_update_map)
        findButton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor)) # Mostra a mãozinha ao passar o mouse
        # Adiciona espaços flexíveis antes e depois para centralizar o botão
        buttonLayout.addStretch(); buttonLayout.addWidget(findButton); buttonLayout.addStretch()

        # --- Montagem Final do Painel ---
        # Adiciona todos os componentes ao layout vertical do painel
        controlPanelLayout.addWidget(titleLabel) # Título
        controlPanelLayout.addLayout(gridLayout) # Grade com rótulos e menus
        controlPanelLayout.addSpacing(20) # Espaço antes do botão
        controlPanelLayout.addLayout(buttonLayout) # Layout com o botão
        controlPanelLayout.addStretch() # Empurra tudo para cima, deixando espaço vazio abaixo

        return controlPanelWidget # Retorna o painel montado

    def _update_destination_options(self) -> None:
        """
        Atualiza as opções disponíveis no menu de Destino baseado na Origem selecionada.
        Também habilita/desabilita o menu de Destino.
        """
        if not self.originCombo or not self.destCombo: return # Segurança, verifica se os widgets existem

        origin_city = self.originCombo.currentText() # Pega a cidade de origem selecionada
        current_dest = self.destCombo.currentText() # Guarda a seleção atual do destino (para tentar manter)

        # Bloqueia sinais temporariamente para evitar que a mudança de itens dispare _update_stop_options prematuramente
        self.destCombo.blockSignals(True)
        self.destCombo.clear() # Limpa as opções antigas do menu de destino

        # Se uma origem válida foi selecionada e temos uma lista de cidades:
        if origin_city and self.all_cities:
            # Cria a lista de cidades disponíveis para destino (todas, exceto a origem)
            available_cities = [""] + [city for city in self.all_cities if city != origin_city]
            self.destCombo.addItems(available_cities) # Adiciona as cidades ao menu
            self.destCombo.setEnabled(True) # Habilita o menu de destino

            # Tenta restaurar a seleção anterior do destino, se ela ainda for válida
            if current_dest in available_cities:
                self.destCombo.setCurrentText(current_dest)
            else:
                # Se a seleção anterior não é mais válida (ou era vazia), seleciona a opção vazia
                self.destCombo.setCurrentIndex(0)
        else:
            # Se a origem está vazia ou não temos cidades, desabilita o destino
            self.destCombo.addItems([""]) # Adiciona apenas a opção vazia
            self.destCombo.setEnabled(False) # Desabilita o menu
            self.destCombo.setCurrentIndex(0) # Garante que a opção vazia está selecionada

        # Desbloqueia os sinais do menu de destino
        self.destCombo.blockSignals(False)

        # Chama manualmente a atualização das paradas DEPOIS de definir o estado do destino.
        # Isso é importante porque se o destino foi resetado para vazio (setCurrentIndex(0)),
        # o sinal 'currentIndexChanged' pode não ser emitido se já estava em 0,
        # então precisamos garantir que as paradas sejam atualizadas.
        self._update_stop_options()


    def _update_stop_options(self) -> None:
        """
        Atualiza as opções e o estado dos menus de Parada em cascata.
        A disponibilidade e as opções de uma parada dependem das seleções anteriores (origem, destino, paradas anteriores).
        """
        if not self.originCombo or not self.destCombo or not self.stopCombos or not self.all_cities: return # Segurança

        origin_city = self.originCombo.currentText()
        dest_city = self.destCombo.currentText()

        # Condição base: As paradas só podem ser habilitadas se Origem E Destino válidos foram selecionados.
        base_stops_enabled = bool(origin_city and dest_city)

        # Cidades que NUNCA podem ser selecionadas como parada (a origem e o destino final)
        base_cities_to_exclude = {origin_city, dest_city}

        # Itera sobre os menus de parada para atualizar cada um
        selected_stops_so_far: Set[str] = set() # Guarda as paradas já selecionadas nas iterações anteriores
        # Flag para controlar a cascata: a próxima parada só pode ser habilitada se a anterior estava habilitada
        # (e, opcionalmente, selecionada, mas como são opcionais, só verificamos a habilitação)
        previous_stop_enabled = True # A primeira parada depende apenas da condição base

        for i, stop_combo in enumerate(self.stopCombos):
            current_selection = stop_combo.currentText() # Guarda a seleção atual desta parada

            # Bloqueia sinais desta parada temporariamente
            stop_combo.blockSignals(True)

            # Determina se esta parada (stop_combo) deve ser habilitada:
            # Precisa da condição base (origem/destino OK) E que a parada anterior estivesse habilitada.
            should_enable = base_stops_enabled and previous_stop_enabled

            stop_combo.setEnabled(should_enable) # Habilita ou desabilita o menu
            stop_combo.clear() # Limpa as opções antigas

            if should_enable:
                # Se o menu está habilitado, calcula as opções disponíveis:
                # Exclui a origem, o destino, E todas as paradas selecionadas ANTERIORMENTE.
                exclude_for_this_combo = base_cities_to_exclude.union(selected_stops_so_far)

                # Cria a lista de cidades disponíveis para esta parada
                available_cities = [""] + [city for city in self.all_cities if city not in exclude_for_this_combo]
                stop_combo.addItems(available_cities) # Adiciona ao menu

                # Tenta restaurar a seleção anterior, se ainda for válida
                if current_selection in available_cities:
                    stop_combo.setCurrentText(current_selection)
                    # Se restaurou uma seleção válida, adiciona ao conjunto para a próxima iteração
                    if current_selection: # Não adiciona a string vazia
                        selected_stops_so_far.add(current_selection)
                    # A próxima parada também pode ser habilitada (continua a cascata)
                    previous_stop_enabled = True
                else:
                    # Se a seleção anterior não é mais válida, volta para a opção vazia
                    stop_combo.setCurrentIndex(0)
                    # Mesmo que esta parada fique vazia, a próxima ainda pode ser habilitada (paradas opcionais)
                    previous_stop_enabled = True

            else: # Se o menu não deve ser habilitado
                stop_combo.addItems([""]) # Adiciona só a opção vazia
                stop_combo.setCurrentIndex(0) # Seleciona a opção vazia
                # Interrompe a cascata de habilitação para as próximas paradas
                previous_stop_enabled = False

            # Desbloqueia os sinais desta parada
            stop_combo.blockSignals(False)

        # Não precisamos disparar atualizações manualmente aqui, a lógica de cascata cuida disso.


    def _create_results_map_panel(self) -> QWidget:
        """Cria o painel direito da interface, com as caixas de texto para resultados/detalhes e a área do mapa."""
        resultsMapWidget = QWidget() # Container do painel
        resultsMapLayout = QVBoxLayout(resultsMapWidget) # Organiza verticalmente

        # Divisor vertical para separar a área de texto (acima) da área do mapa (abaixo)
        resultsSplitter = QSplitter(Qt.Orientation.Vertical)

        # --- Área Superior: Resultados e Detalhes do A* ---
        resultsStepsWidget = QWidget() # Container para os textos
        resultsStepsLayout = QVBoxLayout(resultsStepsWidget) # Organiza os textos verticalmente

        # Rótulo e Caixa de Texto para os Resultados Principais
        resultLabel = QLabel('Resultados do Cálculo:')
        resultLabel.setObjectName("resultLabel") # ID para CSS
        self.resultText = QTextEdit() # Caixa de texto
        self.resultText.setReadOnly(True) # Usuário não pode editar
        self.resultText.setPlaceholderText("Segmentos da rota e custos aparecerão aqui.") # Texto inicial
        self.resultText.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Deixa esticar
        self.resultText.setMinimumHeight(150) # Altura mínima

        # Rótulo e Caixa de Texto para os Detalhes do A*
        stepsLabel = QLabel("Detalhes da Busca A* por Segmento:") # Rótulo atualizado
        stepsLabel.setObjectName("stepsLabel") # ID para CSS
        self.stepsText = QTextEdit() # Caixa de texto
        self.stepsText.setReadOnly(True)
        # Texto inicial explicando o que será mostrado
        self.stepsText.setPlaceholderText(
            "Detalhes da expansão de nós pelo algoritmo A* para cada segmento da rota calculada.\n"
            "Mostra a ordem em que as cidades foram avaliadas, seus custos (f, g, h) e de onde vieram.\n"
            "O algoritmo sempre expande o nó com o menor custo 'f' (custo real 'g' + heurística 'h') da lista aberta."
        )
        # Usa uma fonte monoespaçada (como Consolas ou Courier New) para alinhar melhor o texto do log
        stepsFont = QFont("Consolas", 9)
        if stepsFont.styleHint() != QFont.StyleHint.Monospace: # Se Consolas não for monoespaçada no sistema
             stepsFont = QFont("Courier New", 9) # Tenta Courier New
        self.stepsText.setFont(stepsFont)
        # Permite que esta caixa de texto também estique verticalmente
        self.stepsText.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Removemos a altura máxima para dar mais espaço se necessário

        # Adiciona os rótulos e caixas de texto ao layout superior
        resultsStepsLayout.addWidget(resultLabel)
        resultsStepsLayout.addWidget(self.resultText, 1) # O '1' é um fator de 'stretch'
        resultsStepsLayout.addWidget(stepsLabel)
        resultsStepsLayout.addWidget(self.stepsText, 2) # Damos um 'stretch' maior para os detalhes

        # --- Área Inferior: Mapa (Placeholder) ---
        mapWidget = QWidget() # Container para o mapa
        mapLayout = QVBoxLayout(mapWidget) # Organiza o rótulo e o mapa verticalmente

        mapLabel = QLabel('Mapa da Rota (abre no navegador externo):') # Rótulo atualizado
        mapLabel.setObjectName("mapLabel") # ID para CSS

        # MODIFICAÇÃO: Cria um QLabel descritivo em vez de tentar usar WebEngine
        self.mapView = QLabel("O mapa será exibido no seu navegador padrão quando você calcular uma rota.\n"
                             "Isso contorna problemas de compatibilidade com a renderização interna.")
        self.mapView.setObjectName("mapPlaceholderLabel") # ID para CSS
        self.mapView.setAlignment(Qt.AlignmentFlag.AlignCenter) # Centraliza o texto
        self.mapView.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Deixa esticar
        self.mapView.setMinimumHeight(400) # Altura mínima

        # Adiciona o rótulo do mapa e o widget de visualização (QLabel) ao layout do mapa
        mapLayout.addWidget(mapLabel)
        mapLayout.addWidget(self.mapView)
        print("Debug: QLabel de placeholder do mapa adicionado ao layout.")

        # --- Montagem Final do Painel Direito ---
        # Adiciona a área de texto (superior) e a área do mapa (inferior) ao divisor vertical
        resultsSplitter.addWidget(resultsStepsWidget)
        resultsSplitter.addWidget(mapWidget)
        # Define a proporção entre as áreas (1 parte para texto, 2 para mapa)
        resultsSplitter.setStretchFactor(0, 1)
        resultsSplitter.setStretchFactor(1, 2)
        resultsSplitter.setSizes([350, 450]) # Sugestão de tamanhos iniciais

        # Adiciona o divisor vertical ao layout principal do painel direito
        resultsMapLayout.addWidget(resultsSplitter)

        return resultsMapWidget # Retorna o painel direito montado

    # REMOVIDO: _on_map_load_finished e _on_render_process_terminated não são mais necessários
    # pois não estamos usando QWebEngineView para exibir o mapa.

    def load_initial_map_async(self) -> None:
        """
        Função intermediária para carregar o mapa inicial.
        É chamada pelo QTimer para garantir que a interface já esteja visível.
        """
        print("Debug: load_initial_map_async chamado via QTimer")
        self.create_initial_map() # Chama a função que realmente cria o mapa

    def create_initial_map(self) -> None:
        """
        Cria o mapa inicial e o abre no navegador padrão, em vez de tentar mostrar na aplicação.
        """
        print("Debug: Iniciando create_initial_map")
        if not self.coordinates:
            print("Debug: create_initial_map - Sem coordenadas para mostrar no mapa inicial. Abortando.")
            return

        try:
            print("Debug: Criando mapa Folium inicial...")
            # Define o centro e o nível de zoom inicial para o mapa
            map_center: List[float] = [46.0, 25.0] # Centro aproximado da Romênia
            m = folium.Map(location=map_center, zoom_start=7, tiles='CartoDB positron') # Usando um tema de mapa mais limpo

            # Adiciona marcadores circulares pequenos para todas as cidades como referência
            added_markers = 0
            for city, coord in self.coordinates.items():
                 if coord and len(coord) == 2: # Verifica se a coordenada é válida
                     folium.CircleMarker(
                         location=coord,
                         radius=3, # Raio pequeno
                         color='gray', # Cor cinza
                         fill=True,
                         fill_color='gray',
                         fill_opacity=0.6, # Um pouco transparente
                         tooltip=city # Mostra o nome da cidade ao passar o mouse
                     ).add_to(m) # Adiciona o marcador ao mapa
                     added_markers += 1
            print(f"Debug: {added_markers} marcadores de cidade adicionados ao mapa Folium.")

            # Garante que o diretório temporário onde salvaremos o HTML existe
            temp_dir = os.path.dirname(self.map_file_path)
            if not os.path.exists(temp_dir):
                 os.makedirs(temp_dir)
                 print(f"Debug: Diretório temporário criado: {temp_dir}")

            # Salva o mapa Folium como um arquivo HTML temporário
            print(f"Debug: Salvando mapa inicial em {self.map_file_path}...")
            m.save(self.map_file_path)
            print(f"Debug: Mapa inicial salvo com sucesso.")

            # Verifica se o arquivo foi realmente criado antes de tentar carregá-lo
            if os.path.exists(self.map_file_path):
                # MUDANÇA IMPORTANTE: Abre o mapa no navegador padrão em vez de no QWebEngineView
                url = 'file://' + os.path.abspath(self.map_file_path)
                print(f"Debug: Abrindo mapa no navegador padrão: {url}")
                webbrowser.open(url)

                # Atualiza a interface para mostrar onde o mapa foi aberto
                if self.resultText:
                    self.resultText.append("\n<b>Mapa inicial aberto no seu navegador padrão.</b>")

                if isinstance(self.mapView, QLabel):
                    self.mapView.setText(f"Mapa inicial aberto no navegador padrão.\nCaminho do arquivo: {self.map_file_path}")
            else:
                # Se o arquivo não existe após tentar salvar (problema de permissão?)
                print(f"Erro Crítico: Arquivo do mapa inicial não encontrado após salvar: {self.map_file_path}")
                QMessageBox.critical(self, "Erro de Mapa", f"Falha crítica ao criar o arquivo de mapa temporário:\n{self.map_file_path}\nVerifique as permissões de escrita no diretório temporário.")

        except Exception as e:
            # Captura qualquer erro que possa ocorrer durante a criação/salvamento do Folium ou abertura no navegador
            print(f"Erro CRÍTICO ao criar/salvar ou abrir mapa inicial: {e}")
            traceback.print_exc()
            QMessageBox.warning(self, "Erro de Mapa", f"Não foi possível gerar ou exibir o mapa inicial:\n{e}")
        print("Debug: create_initial_map concluído.")


    def generate_and_display_map(self, segments_data: List[Dict[str, Any]]) -> None:
        """
        Gera um mapa Folium mostrando a rota calculada (com seus segmentos)
        e o abre no navegador padrão.
        """
        print("Debug: Iniciando generate_and_display_map")
        if not self.coordinates:
             if self.resultText: self.resultText.append("\n\nExibição do mapa desabilitada (sem dados de coordenadas).")
             print("Debug: generate_and_display_map - Sem coordenadas.")
             return

        # Se não recebemos dados de segmento (ex: origem=destino sem paradas),
        # apenas mostra o mapa inicial com todas as cidades.
        if not segments_data:
            print("Debug: generate_and_display_map - Sem dados de segmento, chamando create_initial_map.")
            self.create_initial_map()
            return

        # --- Preparação para Desenhar a Rota ---
        all_coords_on_path: List[List[float]] = [] # Lista para guardar as coordenadas de todas as cidades na rota
        valid_segment_found: bool = False # Flag para saber se pelo menos um segmento foi calculado
        cities_on_path: Set[str] = set() # Conjunto para guardar os nomes das cidades na rota

        # Itera sobre os dados dos segmentos recebidos
        for segment in segments_data:
            path: Optional[List[str]] = segment.get('path') # Pega a lista de cidades do segmento
            if path: # Se o segmento tem um caminho válido
                valid_segment_found = True
                for city in path:
                    cities_on_path.add(city) # Adiciona a cidade ao conjunto
                    coord: Optional[List[float]] = self.coordinates.get(city) # Pega as coordenadas da cidade
                    # Adiciona a coordenada à lista se for válida
                    if coord and len(coord) == 2:
                        all_coords_on_path.append(coord)

        # Se não encontramos nenhum segmento válido ou nenhuma coordenada para a rota, mostra o mapa inicial.
        if not valid_segment_found or not all_coords_on_path:
            print("Aviso: Nenhuma coordenada válida encontrada para o caminho calculado. Chamando create_initial_map.")
            self.create_initial_map(); return

        try:
            print("Debug: Criando mapa Folium da rota...")
            # Cria um novo mapa Folium
            m = folium.Map(tiles='CartoDB positron', prefer_canvas=True) # prefer_canvas pode melhorar performance com muitos pontos

            # --- Adiciona Marcadores Base ---
            # Adiciona TODAS as cidades como marcadores cinza de fundo.
            # Destaca um pouco as cidades que fazem parte da rota calculada.
            base_markers = 0
            for city, coord in self.coordinates.items():
                 if coord and len(coord) == 2: # Coordenada válida?
                     is_on_path = city in cities_on_path # Esta cidade está na rota calculada?
                     radius = 5 if is_on_path else 3 # Marcador maior se estiver na rota
                     color = 'darkgray' if is_on_path else 'lightgray' # Cor mais escura se estiver na rota
                     fill_opacity = 0.7 if is_on_path else 0.4 # Mais opaco se estiver na rota

                     folium.CircleMarker(
                         location=coord,
                         radius=radius,
                         color=color,
                         fill=True,
                         fill_color=color,
                         fill_opacity=fill_opacity,
                         tooltip=city # Nome no hover
                     ).add_to(m)
                     base_markers += 1
            print(f"Debug: {base_markers} marcadores base adicionados.")

            # --- Desenha os Segmentos da Rota ---
            # Lista de cores para diferenciar os segmentos (cicla se tiver mais segmentos que cores)
            colors: List[str] = ['blue', 'red', 'green', 'purple', 'orange', 'darkred',
                                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                                 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray',
                                 'black', 'lightgray']
            segment_index: int = 0 # Contador para saber qual segmento estamos desenhando
            special_markers = 0 # Contador de marcadores especiais (origem, destino, paradas)
            polylines = 0 # Contador de linhas de segmento

            # Itera sobre cada segmento da rota calculada
            for segment in segments_data:
                path = segment.get('path')
                if not path: continue # Pula se este segmento não teve caminho encontrado

                segment_coords: List[List[float]] = [] # Coordenadas apenas para este segmento
                current_color: str = colors[segment_index % len(colors)] # Escolhe a cor para este segmento

                # Itera sobre as cidades DENTRO do segmento atual
                for i, city in enumerate(path):
                    coord = self.coordinates.get(city)
                    if coord and len(coord) == 2: # Coordenada válida?
                        segment_coords.append(coord) # Adiciona à lista de coordenadas do segmento

                        # --- Adiciona Marcadores Especiais (Origem, Destino, Paradas) ---
                        icon_name: str = 'circle' # Ícone padrão (não usado se add_marker=False)
                        icon_color = current_color # Cor padrão (não usado se add_marker=False)
                        popup_text = f"<b>{city}</b><br>Segmento: {segment['start']} -> {segment['end']}" # Texto do popup
                        tooltip_text = f"{city}" # Texto do hover
                        add_marker = False # Flag: Adicionar marcador especial neste ponto?

                        # Verifica se é a Origem Geral (primeiro ponto do primeiro segmento)
                        if segment_index == 0 and i == 0:
                            icon_name = 'play' # Ícone de início
                            icon_color = 'green' # Cor verde
                            popup_text = f"<b>{city} (Origem)</b>"
                            add_marker = True
                        # Verifica se é o Destino Final (último ponto do último segmento)
                        elif segment_index == len(segments_data) - 1 and i == len(path) - 1:
                            icon_name = 'stop' # Ícone de parada
                            icon_color = 'red' # Cor vermelha
                            popup_text = f"<b>{city} (Destino)</b>"
                            add_marker = True
                        # Verifica se é uma Parada (primeiro ponto de um segmento que não é o primeiro)
                        elif i == 0 and segment_index > 0:
                             icon_name = 'flag' # Ícone de bandeira
                             icon_color = 'orange' # Cor laranja
                             popup_text = f"<b>{city} (Parada)</b><br>Início: {segment['start']} -> {segment['end']}"
                             add_marker = True

                        # Se for um ponto especial, adiciona um marcador Folium com ícone
                        if add_marker:
                            folium.Marker(
                                location=coord,
                                popup=popup_text, # Texto ao clicar
                                tooltip=tooltip_text, # Texto ao passar o mouse
                                icon=folium.Icon(icon=icon_name, prefix='fa', color=icon_color) # Ícone (Font Awesome)
                            ).add_to(m)
                            special_markers += 1

                # --- Desenha a Linha do Segmento ---
                # Se temos pelo menos 2 pontos no segmento, desenha a linha (PolyLine)
                if len(segment_coords) >= 2:
                    folium.PolyLine(
                        locations=segment_coords, # Lista de coordenadas do segmento
                        color=current_color, # Cor definida para este segmento
                        weight=5, # Espessura da linha
                        opacity=0.8, # Transparência da linha
                        tooltip=f"Segmento {segment_index+1}: {segment['start']} -> {segment['end']} (Custo: {segment['cost']:.1f})" # Info no hover
                    ).add_to(m)
                    polylines += 1

                segment_index += 1 # Vai para o próximo segmento
            print(f"Debug: {special_markers} marcadores especiais e {polylines} polilinhas adicionados.")

            # --- Ajusta o Zoom ---
            # Se temos coordenadas na rota, ajusta o zoom do mapa para mostrar toda a rota.
            if all_coords_on_path:
                 print("Debug: Ajustando limites do mapa (fit_bounds)...")
                 # Calcula os limites geográficos (latitude/longitude mínimas e máximas) da rota
                 # min_lat = min(p[0] for p in all_coords_on_path)
                 # max_lat = max(p[0] for p in all_coords_on_path)
                 # min_lon = min(p[1] for p in all_coords_on_path)
                 # max_lon = max(p[1] for p in all_coords_on_path)
                 # bounds = [[min_lat, min_lon], [max_lat, max_lon]]
                 # Alternativa mais simples: Folium pode calcular os limites automaticamente
                 bounds = m.get_bounds()
                 # Pede ao Folium para ajustar o mapa a esses limites, com uma pequena margem (padding)
                 m.fit_bounds(bounds, padding=(0.1, 0.1)) # Padding de 10%

            # --- Salva e Abre o Mapa Gerado ---
            # Garante que o diretório temporário existe
            temp_dir = os.path.dirname(self.map_file_path)
            if not os.path.exists(temp_dir):
                 os.makedirs(temp_dir)

            print(f"Debug: Salvando mapa da rota em {self.map_file_path}...")
            m.save(self.map_file_path) # Salva o mapa como HTML
            print(f"Debug: Mapa da rota salvo.")

            # Verifica se o arquivo foi salvo e abre no navegador
            if os.path.exists(self.map_file_path):
                # MUDANÇA IMPORTANTE: Abre o mapa no navegador padrão
                url = 'file://' + os.path.abspath(self.map_file_path)
                print(f"Debug: Abrindo mapa da rota no navegador padrão: {url}")
                webbrowser.open(url)

                # Atualiza a interface para mostrar onde o mapa foi aberto
                if self.resultText:
                    self.resultText.append("\n<b>Mapa da rota aberto no seu navegador padrão.</b>")

                if isinstance(self.mapView, QLabel):
                    self.mapView.setText(f"Mapa da rota aberto no navegador padrão.\nCaminho do arquivo: {self.map_file_path}")
            else:
                # Erro se o arquivo não foi encontrado após salvar
                print(f"Erro Crítico: Arquivo do mapa da rota não encontrado após salvar: {self.map_file_path}")
                QMessageBox.critical(self, "Erro de Mapa", f"Falha crítica ao criar o arquivo de mapa da rota:\n{self.map_file_path}")

        except Exception as e:
            # Captura qualquer erro durante a geração do mapa Folium ou abertura no navegador
            print(f"Erro CRÍTICO ao gerar ou abrir mapa da rota: {e}")
            traceback.print_exc()
            QMessageBox.warning(self, "Erro de Mapa", f"Não foi possível gerar ou exibir o mapa da rota:\n{e}")
        print("Debug: generate_and_display_map concluído.")


    def find_path_and_update_map(self) -> None:
        """
        A função principal que é chamada quando o usuário clica no botão "Encontrar Caminho".
        1. Pega as cidades selecionadas (origem, destino, paradas).
        2. Valida as seleções.
        3. Chama o algoritmo A* para cada segmento da rota.
        4. Atualiza as caixas de texto com os resultados e detalhes.
        5. Chama a função para gerar e exibir o mapa com a rota encontrada.
        """
        # --- Verificação Inicial dos Widgets ---
        # Garante que os widgets essenciais da interface foram criados antes de usá-los.
        if not all([self.originCombo, self.destCombo, self.resultText, self.stepsText]):
             print("Erro Interno: Widgets da UI não inicializados corretamente.")
             QMessageBox.critical(self, "Erro Interno", "Componentes da interface não foram criados corretamente.")
             return

        # --- Pega as Seleções do Usuário ---
        start_city: str = self.originCombo.currentText() # Cidade de origem
        end_city: str = self.destCombo.currentText() # Cidade de destino
        # Pega as paradas, ignorando as que foram deixadas em branco
        stops: List[str] = [combo.currentText() for combo in self.stopCombos if combo.currentText()]
        print(f"Debug: Seleção - Origem: '{start_city}', Destino: '{end_city}', Paradas: {stops}")

        # --- Validação da Entrada do Usuário ---
        # Verifica se origem e destino foram selecionados
        if not start_city:
            QMessageBox.warning(self, "Erro de Entrada", "Por favor, selecione uma cidade de origem."); self.originCombo.setFocus(); return
        if not end_city:
            QMessageBox.warning(self, "Erro de Entrada", "Por favor, selecione uma cidade de destino."); self.destCombo.setFocus(); return
        # Verifica se as cidades selecionadas realmente existem na nossa lista de cidades (segurança extra)
        if start_city not in self.all_cities:
             QMessageBox.warning(self, "Erro de Entrada", f"Cidade de origem inválida: '{start_city}'."); return
        if end_city not in self.all_cities:
             QMessageBox.warning(self, "Erro de Entrada", f"Cidade de destino inválida: '{end_city}'."); return
        for i, stop_city in enumerate(stops):
             if stop_city not in self.all_cities:
                 QMessageBox.warning(self, "Erro de Entrada", f"Parada {i+1} inválida: '{stop_city}'."); return

        # --- Validação Lógica da Rota ---
        # Caso simples: Origem e Destino iguais, sem paradas?
        if start_city == end_city and not stops:
             print("Debug: Origem e destino iguais, sem paradas.")
             QMessageBox.information(self, "Resultado", f"Origem e destino são os mesmos: {start_city}\nCusto: 0")
             self.resultText.setText(f"Rota: [{start_city}]\nCusto Total: 0"); self.stepsText.setText("Nenhum cálculo necessário.")
             # Tenta mostrar o mapa com apenas o ponto de origem/destino
             coord = self.coordinates.get(start_city)
             if coord: self.generate_and_display_map([{'path': [start_city], 'start': start_city, 'end': end_city, 'cost': 0}])
             else: self.create_initial_map() # Ou mostra o mapa inicial padrão
             return
        # Verifica se a origem ou destino foram selecionados como parada (a UI deve prevenir, mas verificamos)
        if start_city in stops:
            QMessageBox.warning(self, "Erro de Entrada", f"A cidade de origem ('{start_city}') não pode ser uma parada."); return
        if end_city in stops:
             QMessageBox.warning(self, "Erro de Entrada", f"A cidade de destino ('{end_city}') não pode ser uma parada."); return
        # Verifica se há paradas repetidas
        if len(stops) != len(set(stops)):
             QMessageBox.warning(self, "Erro de Entrada", "Paradas intermediárias não podem ser repetidas."); return

        # Monta a lista completa de pontos da rota na ordem correta: Origem -> Parada1 -> ... -> Destino
        route_points: List[str] = [start_city] + stops + [end_city]
        print(f"Debug: Pontos da rota para cálculo: {route_points}")
        # Verifica se há pontos consecutivos iguais (ex: A -> A) - improvável com a UI atual, mas seguro verificar
        for i in range(len(route_points) - 1):
            if route_points[i] == route_points[i+1]:
                QMessageBox.warning(self, "Erro de Entrada", f"Pontos consecutivos na rota não podem ser iguais: '{route_points[i]}'."); return


        # --- Preparação para o Cálculo ---
        # Limpa as caixas de texto de resultados e detalhes
        self.resultText.clear(); self.stepsText.clear()
        # Mostra uma mensagem de "calculando..."
        self.resultText.setPlaceholderText(f"Calculando rota: {' -> '.join(route_points)}...")
        self.resultText.append("Iniciando busca de caminho...\n")
        print("Debug: Iniciando cálculo A* sequencial...")
        QCoreApplication.processEvents() # Força a interface a se atualizar agora

        # --- Executa o A* Sequencialmente para Cada Segmento da Rota ---
        total_cost: float = 0.0 # Custo acumulado da rota completa
        full_path: List[str] = [] # Lista com todas as cidades da rota completa
        segments_data: List[Dict[str, Any]] = [] # Lista para guardar os dados de cada segmento (para o mapa)
        calculation_successful: bool = True # Flag para saber se todos os segmentos foram calculados com sucesso
        # Dicionário para guardar os detalhes da busca A* de TODOS os segmentos
        all_segments_expanded_nodes_details: Dict[str, List[str]] = {}

        current_start: str = start_city # Começa na origem geral
        segment_label = "" # Rótulo para identificar o segmento (ex: "Segmento 1: Arad -> Sibiu")

        # Itera sobre os pontos da rota para calcular cada segmento (ponto atual -> próximo ponto)
        for i, current_end in enumerate(route_points[1:]): # Começa do segundo ponto (o primeiro destino)
            segment_start_city: str = current_start # O início deste segmento é o fim do anterior (ou a origem geral)
            segment_end_city: str = current_end # O fim deste segmento é o ponto atual do loop
            segment_label = f"Segmento {i+1}: {segment_start_city} -> {segment_end_city}"

            # Atualiza a caixa de resultados indicando qual segmento está sendo calculado
            self.resultText.append(f"<b>Calculando {segment_label}...</b>")
            QCoreApplication.processEvents() # Atualiza a interface
            print(f"Debug: Chamando A* para {segment_start_city} -> {segment_end_city}")

            # Variáveis para guardar o resultado do A* para este segmento
            path_segment: Optional[List[str]] = None
            cost_segment: float = float('inf')
            expanded_nodes_details: List[str] = [] # Log do A* para este segmento

            try:
                # Chama a função A* para encontrar o caminho e o custo deste segmento
                # Passa True para usar a heurística dinâmica (baseada em coordenadas)
                path_segment, cost_segment, expanded_nodes_details = astar(
                    self.graph, self.heuristics, self.coordinates, segment_start_city, segment_end_city, True
                )
                # Guarda os detalhes da busca A* para este segmento
                all_segments_expanded_nodes_details[segment_label] = expanded_nodes_details

            except Exception as e:
                 # Captura erros inesperados durante a execução do A*
                 print(f"Erro CRÍTICO durante A* para {segment_label}: {e}")
                 traceback.print_exc()
                 self.resultText.append(f"<font color='red'>Erro inesperado ao calcular {segment_label}: {e}</font>")
                 calculation_successful = False # Marca que o cálculo falhou
                 break # Interrompe o cálculo dos próximos segmentos

            # --- Processa o Resultado do Segmento ---
            if path_segment: # Se o A* encontrou um caminho para este segmento
                print(f"Debug: Segmento encontrado: {path_segment}, Custo: {cost_segment:.2f}")
                self.resultText.append(f"- Caminho: {' -> '.join(path_segment)}")
                self.resultText.append(f"- Custo do Segmento: {cost_segment:.2f}\n")
                total_cost += cost_segment # Adiciona o custo do segmento ao custo total

                # Adiciona as cidades do segmento ao caminho completo, evitando duplicatas no início/fim dos segmentos
                if not full_path: # Se for o primeiro segmento
                    full_path.extend(path_segment)
                else: # Para segmentos subsequentes, adiciona a partir da segunda cidade (para não repetir a conexão)
                    full_path.extend(path_segment[1:])

                # Guarda os dados deste segmento para usar na geração do mapa
                segments_data.append({
                    'start': segment_start_city,
                    'end': segment_end_city,
                    'path': path_segment,
                    'cost': cost_segment
                })
                # O início do próximo segmento será o fim deste
                current_start = segment_end_city
            else: # Se o A* NÃO encontrou um caminho para este segmento
                print(f"Erro: Caminho não encontrado para {segment_label}")
                self.resultText.append(f"<font color='red'>- Caminho não encontrado para {segment_label}.</font>\n")
                calculation_successful = False # Marca que o cálculo falhou
                break # Interrompe o cálculo dos próximos segmentos

            QCoreApplication.processEvents() # Atualiza a interface entre os segmentos

        # --- Exibe Resultados Finais e Detalhes A* ---
        print("Debug: Cálculo A* sequencial concluído.")
        self.resultText.append("<hr>") # Adiciona uma linha separadora nos resultados

        # Monta o texto final na caixa de resultados
        if calculation_successful: # Se todos os segmentos foram encontrados
            print(f"Debug: Rota completa encontrada. Custo total: {total_cost:.2f}")
            self.resultText.append("<b>Resultado Final da Rota Completa:</b>")
            self.resultText.append(f"- A rota de <b>{route_points[0]}</b> até <b>{route_points[-1]}</b>")
            if stops: self.resultText.append(f"  passando por: <b>{', '.join(stops)}</b>")
            else: self.resultText.append("  (sem paradas)")
            self.resultText.append(f"- Caminho Detalhado: {' -> '.join(full_path)}")
            self.resultText.append(f"- Custo Total Acumulado: <b>{total_cost:.2f}</b>")
        else: # Se algum segmento falhou
            print("Debug: Cálculo da rota incompleto devido a erros.")
            self.resultText.append("<font color='red'><b>Cálculo da rota interrompido.</b></font>")
            self.resultText.append("- Não foi possível encontrar um caminho para todos os segmentos solicitados.")

        # Monta e exibe os detalhes da busca A* para TODOS os segmentos na caixa de detalhes
        steps_output_lines = [] # Lista para guardar as linhas de texto HTML
        # Adiciona uma explicação geral do A*
        steps_output_lines.append("<b>Explicação do Processo A*:</b>")
        steps_output_lines.append("O algoritmo A* busca o caminho de menor custo.")
        steps_output_lines.append("1. Ele mantém uma 'lista aberta' (fila de prioridade) de cidades a visitar, ordenada pelo custo total estimado 'f'.")
        steps_output_lines.append("2. Custo <b>f = g + h</b>, onde:")
        steps_output_lines.append("   - <b>g</b>: Custo <u>real</u> acumulado desde a origem do segmento atual até a cidade.")
        steps_output_lines.append("   - <b>h</b>: Custo <u>estimado</u> (heurística) da cidade até o destino do segmento atual.")
        steps_output_lines.append("3. Em cada passo, a cidade com o <b>menor 'f'</b> na lista aberta é escolhida e 'expandida' (removida da lista aberta e seus vizinhos são avaliados).")
        steps_output_lines.append("4. Vizinhos são adicionados ou atualizados na lista aberta se um caminho melhor for encontrado através do nó expandido.")
        steps_output_lines.append("5. O processo repete até encontrar o destino do segmento ou a lista aberta ficar vazia (sem caminho).")
        steps_output_lines.append("<hr>") # Linha separadora
        steps_output_lines.append("<b>Detalhes da Expansão por Segmento:</b><br>") # Título para os detalhes

        # Verifica se temos algum detalhe para mostrar
        if not all_segments_expanded_nodes_details:
            steps_output_lines.append("Nenhum segmento foi calculado ou nenhum nó foi expandido.")
        else:
            # Itera sobre os detalhes guardados para cada segmento
            for seg_label, details in all_segments_expanded_nodes_details.items():
                steps_output_lines.append(f"<u>{seg_label}:</u>") # Sublinha o título do segmento
                if details:
                    # Adiciona cada linha de detalhe, escapando caracteres HTML para segurança
                    steps_output_lines.extend([line.replace('<', '&lt;').replace('>', '&gt;') for line in details])
                else:
                    steps_output_lines.append(" (Sem detalhes de expansão disponíveis)")
                steps_output_lines.append("") # Adiciona uma linha em branco entre segmentos

        # Define o conteúdo HTML da caixa de texto de detalhes, juntando as linhas com <br> (quebra de linha HTML)
        self.stepsText.setHtml("<br>".join(steps_output_lines))

        # --- Atualiza o Mapa ---
        print("Debug: Chamando generate_and_display_map para abrir a rota no navegador.")
        # Chama a função para gerar o mapa com os dados dos segmentos encontrados e abrir no navegador
        self.generate_and_display_map(segments_data)
        # Restaura o texto placeholder da caixa de resultados
        self.resultText.setPlaceholderText("Segmentos da rota e custos aparecerão aqui.")
        print("Debug: find_path_and_update_map concluído.")

    def closeEvent(self, event) -> None:
        """
        Função chamada automaticamente quando o usuário tenta fechar a janela.
        Usamos para limpar recursos, como deletar o arquivo HTML temporário.
        """
        print("Fechando aplicação (closeEvent)...")
        try:
            # REMOVIDO: Parar o WebEngine não é mais necessário
            # if isinstance(self.mapView, QWebEngineView):
            #      print("Debug: Parando QWebEngineView...")
            #      self.mapView.stop()
            #      self.mapView.setUrl(QUrl("about:blank")) # Limpa a URL

            # Tenta remover o arquivo HTML temporário do mapa
            if self.map_file_path and os.path.exists(self.map_file_path):
                print(f"Debug: Removendo arquivo de mapa temporário: {self.map_file_path}")
                os.remove(self.map_file_path)
        except Exception as e:
            # Se ocorrer um erro durante a limpeza, apenas avisa no console, mas não impede o fechamento.
            print(f"Aviso: Não foi possível remover o arquivo temporário durante closeEvent: {e}")
        finally:
            # Confirma que o evento de fechamento pode prosseguir (a janela pode fechar).
            print("Debug: Aceitando closeEvent.")
            event.accept()


# --- Função para Carregar Dados ---
def load_data(filename: str = "data.json") -> Optional[Dict[str, Any]]:
    """
    Carrega os dados do grafo (conexões, heurísticas, coordenadas) de um arquivo JSON.
    Realiza validações básicas para garantir que o arquivo existe e tem as chaves principais.
    Retorna um dicionário com os dados se tudo estiver OK, ou None se ocorrer um erro crítico.
    """
    # Encontra o caminho absoluto para o arquivo JSON (espera-se que esteja na mesma pasta do script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    print(f"Tentando carregar dados de: {file_path}")

    # Verifica se o arquivo realmente existe
    if not os.path.exists(file_path):
        print(f"Erro Crítico: Arquivo de dados '{filename}' não encontrado em '{file_path}'.")
        return None # Erro fatal

    try:
        # Abre o arquivo JSON para leitura (usando UTF-8 para suportar acentos)
        with open(file_path, 'r', encoding='utf-8') as f:
            # Tenta carregar o conteúdo do arquivo como um dicionário Python
            data: Dict[str, Any] = json.load(f)

            # --- Validação Mínima Essencial ---
            # Verifica se as chaves principais ('connections', 'heuristics', 'coordinates') existem
            required_keys: List[str] = ['connections', 'heuristics', 'coordinates']
            missing_keys: List[str] = [key for key in required_keys if key not in data]
            if missing_keys:
                print(f"Erro Crítico: Arquivo JSON '{filename}' não contém as chaves obrigatórias: {', '.join(missing_keys)}.")
                return None # Erro fatal

            # Verifica se os valores dessas chaves são dicionários (estrutura básica esperada)
            if not isinstance(data.get('connections'), dict) or \
               not isinstance(data.get('heuristics'), dict) or \
               not isinstance(data.get('coordinates'), dict):
                 print(f"Erro Crítico: Arquivo JSON '{filename}' tem formato inválido. 'connections', 'heuristics' e 'coordinates' devem ser dicionários.")
                 return None # Erro fatal

            # Se passou pelas validações mínimas, retorna os dados carregados.
            print("Dados carregados com sucesso.")
            return data

    except json.JSONDecodeError as e:
        # Se o arquivo não for um JSON válido
        print(f"Erro Crítico: Erro ao decodificar JSON de '{filename}'. Verifique o formato.\nDetalhes: {e}")
        return None # Indica falha crítica
    except Exception as e:
        # Captura qualquer outro erro inesperado durante a leitura/processamento do arquivo
        print(f"Erro Crítico: Ocorreu um erro inesperado ao carregar os dados: {type(e).__name__}: {e}")
        traceback.print_exc() # Imprime detalhes do erro para depuração
        return None # Indica falha crítica


# --- Ponto de Entrada Principal da Aplicação ---
# Este bloco só é executado quando o script é rodado diretamente (não importado como módulo).
if __name__ == '__main__':
    print("Debug: Aplicação iniciada.")
    # --- Opções para Debug de Gráficos (OpenGL/ANGLE) ---
    # A opção 'software' já foi definida no início do script.
    # Manter os comentários aqui para referência futura.
    # os.environ['QT_OPENGL'] = 'angle' # Força ANGLE (usa DirectX no Windows, geralmente mais estável)
    # os.environ['QT_ANGLE_PLATFORM'] = 'd3d11' # Especifica DirectX 11 para ANGLE (pode tentar 'd3d9')
    # os.environ['QT_OPENGL'] = 'desktop' # Força usar o OpenGL nativo da placa de vídeo (pode ser instável)
    # os.environ['QT_OPENGL'] = 'software' # Força renderização por software (CPU, lento, mas bom para teste)
    print(f"Debug: QT_OPENGL={os.environ.get('QT_OPENGL')}") # Mostra qual opção está ativa (se alguma)
    print(f"Debug: QT_ANGLE_PLATFORM={os.environ.get('QT_ANGLE_PLATFORM')}")

    # --- 1. Instanciar QApplication PRIMEIRO ---
    # Isso é fundamental no PyQt: criar o objeto QApplication antes de qualquer widget.
    print("Debug: Criando QApplication...")
    app = QApplication(sys.argv) # 'sys.argv' permite passar argumentos de linha de comando (geralmente não usado aqui)
    print("Debug: QApplication criada.")

    # --- 2. Carregar Dados do Grafo ---
    print("Debug: Carregando dados do grafo...")
    # Chama a função para carregar e validar minimamente o arquivo data.json
    graph_data: Optional[Dict[str, Any]] = load_data("data.json")

    # --- 3. Verificar Dados e Criar a Janela Principal ---
    if graph_data: # Procede somente se os dados foram carregados com sucesso (load_data não retornou None)
        print("Debug: Dados carregados, criando janela principal (AStarApp)...")
        main_window: Optional[AStarApp] = None # Variável para guardar a janela principal
        try:
            # Cria a instância da nossa classe AStarApp, passando os dados carregados
            main_window = AStarApp(graph_data)
            print("Debug: Instância AStarApp criada.")

            # Torna a janela visível na tela
            print("Debug: Mostrando janela principal (main_window.show())...")
            main_window.show()
            print("Debug: main_window.show() chamado.")

            # --- Carrega o Mapa Inicial (Assíncrono) ---
            # Carregamos o mapa inicial *depois* que a janela já está visível.
            # Usamos QTimer.singleShot para agendar a execução da função 'load_initial_map_async'
            # um pouquinho depois (100ms), garantindo que o loop de eventos do Qt já esteja rodando.
            print("Debug: Agendando carregamento/abertura do mapa inicial com QTimer.singleShot...")
            QTimer.singleShot(100, main_window.load_initial_map_async)

            # --- 4. Iniciar o Loop de Eventos Principal ---
            # Entrega o controle da execução para o Qt. A partir daqui, o Qt gerencia
            # a interface, esperando por ações do usuário (cliques, seleções) e
            # processando eventos. O código só continua após o fechamento da janela.
            print("Debug: Iniciando loop de eventos principal (app.exec())...")
            exit_code = app.exec() # Bloqueia até a janela ser fechada
            print(f"Debug: Loop de eventos terminado com código: {exit_code}")
            # Encerra o script Python com o mesmo código de saída que o Qt retornou.
            sys.exit(exit_code)

        # --- Tratamento de Erros Durante a Inicialização ou Execução ---
        except ValueError as ve:
             # Captura o erro específico que lançamos em AStarApp.__init__ se a validação dos dados falhar.
             print(f"Erro de Inicialização da Aplicação: {ve}")
             traceback.print_exc() # Mostra detalhes do erro no console
             # Mostra uma mensagem de erro para o usuário
             QMessageBox.critical(None, "Erro de Inicialização", f"Falha ao inicializar a aplicação:\n{ve}")
             sys.exit(1) # Encerra o script com um código de erro
        except Exception as e:
             # Captura qualquer outro erro inesperado que possa ocorrer durante a criação da janela ou no loop de eventos.
             print(f"Erro de Execução Não Tratado: {type(e).__name__} - {e}")
             traceback.print_exc() # Mostra detalhes do erro no console
             # Mostra uma mensagem de erro grave para o usuário
             QMessageBox.critical(None, "Erro Crítico de Execução",
                                  f"Ocorreu um erro inesperado e a aplicação precisa fechar:\n"
                                  f"{type(e).__name__}: {e}\n\n"
                                  "Verifique o console para mais detalhes técnicos.")
             sys.exit(1) # Encerra o script com um código de erro
    else:
        # Se load_data retornou None (erro crítico ao carregar/validar data.json)
        # Mostra uma mensagem de erro fatal antes mesmo de tentar criar a janela.
        QMessageBox.critical(None, "Erro Fatal nos Dados",
                             "Não foi possível carregar ou validar os dados essenciais do arquivo 'data.json'.\n"
                             "A aplicação não pode continuar.\n"
                             "Verifique o console para detalhes do erro.")
        print("Saindo devido a falha crítica no carregamento dos dados.")
        sys.exit(1) # Encerra o script com um código de erro