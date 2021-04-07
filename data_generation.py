import pandas as pd
import numpy as np
from pprint import pprint
import random
import networkx as nx 
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET
import collections
import os
import subprocess
import bs4

# deprecation warnings
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


body_constr = [0,20]
head_constr = [0,10]
max_root_time = 100000 - head_constr[1]

# Constants
HEAD_CONST = 'head'
ROOT_CONST = 'r'
A = 'a'
always = 100
cycle_prob = 50

class GraphVisualization: 
   
    def __init__(self): 
          
        # visual is a list which stores all  
        # the set of edges that constitutes a 
        # graph 
        self.visual = [] 
          
    # addEdge function inputs the vertices of an 
    # edge and appends it to the visual list 
    def addEdge(self, a, b): 
        temp = [a, b] 
        self.visual.append(temp) 
          
    # In visualize function G is an object of 
    # class Graph given by networkx G.add_edges_from(visual) 
    # creates a graph with a given list 
    # nx.draw_networkx(G) - plots the graph 
    # plt.show() - displays the graph 
    def visualize(self): 
        G = nx.Graph() 
        G.add_edges_from(self.visual) 
        nx.draw_networkx(G) 
        plt.show()

    def save_graph(self, path):
        G = nx.Graph() 
        G.add_edges_from(self.visual) 
        nx.draw_networkx(G)
        plt.savefig(path) 


def time_interval(body_constr): return tuple(np.random.uniform(body_constr)) 

def unique_body_symbols(low=3, high=5, single=False):
    '''Creates a random unique list of symbols for the body 
    of pattern. Symbol A has been excluded from the list as it is reserved 
    for the head of the pattern.
        Args: 
            n_body_symbols (int): amount of body symbols
            
        Returns:
            random_symbols (list): random choice of symbols
    '''
    symbols = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'] # possible body symbols
    n_bod_symbols = np.random.randint(low, high)
    if single:  n_bod_symbols = 1 
    rand_symbols = random.sample(symbols, k=n_bod_symbols)
    
    return rand_symbols

def make_edges(Edge, body_symbols, condition=None, connected_nodes=None):
    '''Args:
            Edge (named_tuple): Edge object with default fields.
            body_symbols (list): unique body symbols.
            condition (str/logical???): Conditional operator, if any.
            connected_nodes (list): list of currently connected nodes to 
            graph object.
        Returns:
            edges (list of named_tuples): 
            connected_nodes (list): 
            
    '''
    
    if condition == 'conjunction' or condition == 'disjunction':
        n_body_symbols = 2
    elif condition == 'negation' or condition == 'cycle':
        n_body_symbols = 1
    elif condition == None:
        n_body_symbols = len(body_symbols)
    
    edges = []
    if connected_nodes is None: connected_nodes = []
    
    for _ in range(n_body_symbols):
        sym = body_symbols.pop()
        start_t, end_t = time_interval(body_constr)
        rand_end_s = np.random.choice(connected_nodes + [ROOT_CONST])        

        if condition == 'disjunction':
            ### HANDLE Low > High error
            edges.append(Edge(ROOT_CONST, sym, start_t, end_t, prob=always, disjunction=True))
            connected_nodes.append(sym)
        elif condition == 'conjunction':
            edges.append(Edge(ROOT_CONST, sym, start_t, end_t, prob=always, conjunction=True))
            connected_nodes.append(sym)
        elif condition == 'negation':
            edges.append(Edge(rand_end_s, sym, start_t, end_t, prob=always, negation=True))
            connected_nodes.append(sym)
        elif condition == 'cycle':
            edges.append(Edge(rand_end_s, sym, start_t, end_t, prob=cycle_prob, cycle=True))
            connected_nodes.append(sym)
        # no conditional operator, append edge to connected nodes or root
        elif condition == None:
            edges.append(Edge(rand_end_s, sym, start_t, end_t, prob=always))
            connected_nodes.append(sym)

    return edges, connected_nodes

def body_pattern(low_body=4, high_body=6, low_prob=60, high_prob=90, disjunction=False,
                   negation=False, conjunction=False, prob=always, cycle=False):
    
    body_symbols = unique_body_symbols(low_body, high_body) # technically, body_symbols excluding root
    rand_num = np.random.uniform(low=0, high=always)
    prob = np.random.uniform(low=low_prob, high=high_prob)
    connected_nodes = [ROOT_CONST] # keep track of connected nodes for end_symbol possibilities
    
    pattern = [] # pattern list of edges : graph like object
    fields = ('start_s', 'end_s', 'start_t', 'end_t', 'disjunction', 'negation', 'conjunction', 'cycle', 'prob')
    Edge = collections.namedtuple('Edge', fields)
    Edge.__new__.__defaults__ = (False,) * len(Edge._fields) # set default fields

        
    if disjunction:
        
        max_disjs = len(body_symbols) // 2
        n_disjs = np.random.randint(low=1, high=max_disjs)
        for disj in range(n_disjs): 
            edges, nodes = make_edges(Edge, body_symbols, condition='disjunction')
            pattern.append(Edge(nodes[0], nodes[1], # disjunction edge
                start_t=None, end_t=None, disjunction=True, prob=always))
            pattern.extend(edges)
            connected_nodes.extend(nodes)

    if cycle:
        edges, nodes = make_edges(Edge, body_symbols, condition='cycle') # create cycle edge
        pattern.append(Edge(nodes[0], np.random.choice(connected_nodes),
                           disjunction=True)) # create disjunction edge to stop infinite loop
        pattern.extend(edges)
        connected_nodes.extend(nodes) 
        
    elif negation:
        edges, nodes = make_edges(Edge, body_symbols, condition='negation',
                                 connected_nodes=connected_nodes)
        pattern.extend(edges)
        connected_nodes.extend(nodes) 
        
    elif conjunction:
        edges, nodes = make_edges(Edge, body_symbols, condition='conjunction')
        pattern.extend(edges)
        connected_nodes.extend(nodes)
    # remaining nodes are added to either root or connected body node
    edges, nodes = make_edges(Edge, body_symbols, condition=None,
                             connected_nodes=connected_nodes)
    pattern.extend(edges)
    connected_nodes.extend(nodes)

    return pattern

def plot_pattern(pattern, show=True, save=False, path=None): # sometimes returns None? unsure why. annoying
    G = GraphVisualization() 
    # print(pattern)
    for p in pattern: 
        G.addEdge(p.start_s, p.end_s) 
    
    if show: G.visualize()

    if save: G.save_graph(path)
    

def time_point(start_t, end_t): return np.random.uniform(start_t, end_t)

def generate_edge_instance(tp,edge):
    return tp - time_point(edge.start_t, edge.end_t), edge.end_s

def outgoing_edges(node, pattern):
    '''Find immediate edges for a node'''
    out_edges = []
    for edge in pattern:            
        if edge.disjunction and edge.start_t is None: continue
            
        elif edge.negation == True: continue # skip negation node        
            
        elif edge.start_s == node:            
            out_edges.append(edge)

    return out_edges

def dest_nodes(node,pattern):
    out_edges = outgoing_edges(node,patt)
    return [edge.end_s for edge in out_edges]
    
def mutual_excl_edges(pattern):
    '''Find all mutually exclusive edges in pattern'''
    return [e for e in pattern if e.start_t == None]

def have_mutual(node, excl_edges):
    '''Is a given node connected to a mutually exclusive node'''
    for edge in excl_edges:
        if node == edge.start_s or node == edge.end_s:
            return True
        
    else: False
        
def generate_neighbouring_tps(node,tp,patt):
    out_edges = outgoing_edges(node,patt)
    excl_edges = mutual_excl_edges(patt)

    # determine which edge to choose from each mutual exclusion
    skip_nodes = []
    for edge in excl_edges:
        if np.random.randint(2) == 1:
            skip_nodes.append(edge.start_s)
        else:
            skip_nodes.append(edge.end_s)

    tps = []
    for edge in out_edges:
        if edge.prob != 100 and rand_num > 50: continue
        elif edge.end_s in skip_nodes: continue
        tps.append(generate_edge_instance(tp,edge))
        
    return tps
        
        
def generate_tps(node,tp,patt):
    all_tps = []

    if node == None:
        # this is the start of recursive generation, start from root
        node = ROOT_CONST
        tp = np.random.uniform(low=0, high=max_root_time)
        all_tps.append((tp, node))
    
    neighbouring_tps = generate_neighbouring_tps(node,tp,patt)
    all_tps += neighbouring_tps
    
    for neighbouring_tp in neighbouring_tps:
        # recursive step (because calling this same method)
       all_tps += generate_tps(neighbouring_tp[1],neighbouring_tp[0],patt)
    
    return all_tps

def generate_event_pred(patt, head_prob, make_pred=False):
    '''
    For event instance, only add A if head prob is exceeded
    Input: 
    Returns: 
    '''    
    body_patt = patt[0]
    head_patt = patt[1]
    
    if make_pred == False:
        body_patt = rand_subset(patt[0])
    
    body_inst = generate_tps(None, None, body_patt)

    pred = []
    
    for t in body_inst: # get root tp
        if t[1] == ROOT_CONST: 
            root_tp = t[0]
        
    head_tp = root_tp + time_point(head_patt[1], head_patt[2])    
    consequent = (head_tp, A)
    pred.append(consequent)
        
    if head_prob > np.random.randint(low=0, high=100):
        body_inst.append(consequent) 

    if make_pred:
        return body_inst, pred

    else:
        return body_inst

def rand_subset(body_patt): # random subset of the instance
    '''
    Returns:
        ts (list): list of lists of time point and symbol of
        random subset of the pattern definition'''
    if len(body_patt) == 1: return []
    
    subset_idxs = sorted([np.random.randint(low=0, high=len(body_patt)) for _ in range(2)])
    sub_pattern = body_patt[subset_idxs[0]: subset_idxs[1]]
    
    return sub_pattern
    
def noisy_instance(time_high=100000):
    ''' Input: 
            time_high (int): 
    
            Returns:
                ts (list): list of lists containing random time point and symbol'''
    
    sym = unique_body_symbols(single=True).pop()
    tp = time_point(start_t=0, end_t=time_high)
    inst = [[tp, sym]]
    return inst

def generate_events_preds(pattern, head_prob, n_patterns, n_subsets, n_noisy_insts):
    '''Creates both .event array and its corresponding .pred array from
    
    
    Returns:
            events (list): time series for training/testing set
            preds (list): ground truth values
    '''    
    events, preds= [], []
    
    for _ in range(n_patterns):
        event, pred =  generate_event_pred(pattern, head_prob, make_pred=True)
        events.extend(event)
        preds.extend(pred)
        
    for _ in range(n_subsets):
        event = generate_event_pred(pattern, head_prob, make_pred=False) # dont think head prob needed
        if event == None:
            continue
        elif event != []:
            events.extend(event)
        else:
            continue
        
    for _ in range(n_noisy_insts):
        event = noisy_instance()
        events.extend(event)
        

    events.sort(key=lambda x: x[0])
    preds.sort(key=lambda x: x[0])# sort by timestamp
    
    return events, preds

ROOT_DIR = 'C:/Users/admin/Documents/TemporalDataGeneration/'

def make_event_dir():
    os.chdir(f"{ROOT_DIR}/data/")

    event_id = 0
    while os.path.exists(f"./pattern_{event_id}"):
        event_id += 1

    new_dir = f"../data/pattern_{event_id}"

    os.mkdir(new_dir)    
    os.chdir(new_dir)
    
    return event_id

def array_to_file(file_name, array, log=False):
    with open(file_name, 'w') as file:
        for l in array:
            if log:
                file.write(str(l) + '\n') 
            else:
                s = str(l[0]) + " " + str(l[1]) + '\n'
                file.write(s)
                
def make_learning_config(pattern_path, event_id):
    xml = ET.parse(f'{ROOT_DIR}/titarl_learning_config.xml')
    root = xml.getroot()
    
    for child in root:
        if child.get('name') == 'saveRules_new':
            ## WEIRD UNICODE BUG? sometimes outputs pattern_14&#10;ew_rules_14.xml
            print(child.set('value', f'{pattern_path}\\new_rules_{event_id}.xml'))

        if child.get('path'):
            child.set('path', f'{pattern_path}\pattern_{event_id}_1.evt') ### MAKE FOR TRAINING SET???
        
    xml.write(f'./pattern_{event_id}_learning_config.xml')
    

def make_patt_files(pattern, events1, preds1, events2, preds2, pattern_path, event_id):
    
    array_to_file(f'{pattern_path}\pattern_{event_id}.log', pattern, log=True) # log the pattern
    
    array_to_file(f'{pattern_path}\pattern_{event_id}_1.evt', events1)
    array_to_file(f'{pattern_path}\pattern_{event_id}_1.pred', preds1) # write preds to .pred file

    
    array_to_file(f'{pattern_path}\pattern_{event_id}_2.evt', events2)
    array_to_file(f'{pattern_path}\pattern_{event_id}_2.pred', preds2) # write preds to .pred file
    
    make_learning_config(pattern_path, event_id)

def make_pattern(body_patt):
    start_t, end_t = time_interval(head_constr)
    consequent = tuple([A, start_t, end_t])

    return (body_patt, consequent)

def generate_pattern_outputs(patt, body_patt, head_prob, pattern_path, event_id, n_patterns,
                            n_subsets, n_noisy_insts):
    
    # save .png graph
    plot_pattern(body_patt, show=False, save=True, path=f'{pattern_path}\pattern_{event_id}.png')

    events1, preds1 = generate_events_preds(patt, head_prob, n_patterns, 
                                            n_subsets, n_noisy_insts)
    
    events2, preds2 = generate_events_preds(patt, head_prob, n_patterns, 
                                            n_subsets, n_noisy_insts)

    make_patt_files(patt, events1, preds1, events2, preds2, 
                    pattern_path=pattern_path, event_id=event_id)


def run_titarl(titarl_path, pattern_path, event_id):
    learn_path = f'{pattern_path}\pattern_{event_id}_learning_config.xml'
    display_path = fr'{pattern_path}\new_rules_{event_id}.xml'
    output_path = fr'{pattern_path}\new_rules_{event_id}.html'

    cmd = [titarl_path, '--learn', learn_path, '--display', display_path,
            '--output', output_path] 
    print(subprocess.run(cmd, shell=True))

def foo():
    n_patterns = 1000
    n_subsets = 5000
    n_noisy_insts = 0
    head_prob = 60

    body_patt = body_pattern(low_body=1, high_body=2)
    patt = make_pattern(body_patt)

    root_path = os.getcwd()
    titarl_path = root_path + '\TITARL.exe'
    event_id = make_event_dir()
    pattern_path = root_path + f'\data\pattern_{event_id}'

    generate_pattern_outputs(patt, body_patt, head_prob, pattern_path, event_id, n_patterns,
                            n_subsets, n_noisy_insts)
    run_titarl(titarl_path, pattern_path, event_id)

foo()

    # run TITARL.exe



