import networkx as nx
import numpy as np
from scipy.special import erfc, erf
import random



def parse_bench_file(file_path):
    # Create a directed graph
    graph = nx.DiGraph()

    # Open and read the .bench file
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('INPUT'):
                input_node = line.split('(')[1].strip(')')
                graph.add_node(input_node, type='input')
            elif line.startswith('OUTPUT'):
                output_node = line.split('(')[1].strip(')')
                graph.add_node(output_node, type='output')
            else:
                output, expression = line.split('=')
                output = output.strip()
                gate_type = expression.split('(')[0].strip()
                inputs = expression.split('(')[1].strip(')').split(',')

                for input_node in inputs:
                    input_node = input_node.strip()
                    graph.add_edge(input_node, output, gate=gate_type)

    return graph

def print_graph_info(graph):
    print("Nodes:")
    for node in graph.nodes(data=True):
        print(node)

    print("\nEdges:")
    for edge in graph.edges(data=True):
        print(edge)

def find_all_io_paths(graph):
    input_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'input']
    output_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'output']

    all_paths = []

    for input_node in input_nodes:
        for output_node in output_nodes:
            paths = list(nx.all_simple_paths(graph, source=input_node, target=output_node))
            all_paths.extend(paths)

    return all_paths


def get_gate_info(graph, node1, node2):
    if graph.has_edge(node1, node2):
        return graph[node1][node2]['gate']
    else:
        return "No such edge exists"

#graph = parse_bench_file(circuit+'.bench')
#paths = find_all_io_paths(graph)


#print_graph_info(graph)
#get_gate_info(graph, 'G1', 'G10')

def parse_time_file(file_path):
    # Dictionary to store the timing information for each edge
    edge_timings = {}

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                continue  # Skip any malformed lines

            start_wire = parts[0]
            stop_wire = parts[1]
            timings = list(map(float, parts[2:]))  # Convert timing values to floats

            # Store the timing information, using a tuple of (start_wire, stop_wire) as the key
            edge_timings[(start_wire, stop_wire)] = timings

    return edge_timings

def print_timing_info(edge_timings):
    for edge, timings in edge_timings.items():
        print(f"Edge from {edge[0]} to {edge[1]}: Timing values - {timings}")


    # Replace 'path_to_file.time' with the path to your actual .time file
#interconnect_timings = parse_time_file(circuit+'.time')
#print(interconnect_timings)

def parse_gate_time_file(file_path):
    gate_info = {}
    current_gate = None

    with open(file_path, 'r') as file:
        for line in file:
            # Clean up the line and skip empty or comment lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Identify gate definition
            if line.startswith('GATE:'):
                current_gate = line.split(':')[1].strip()
                gate_info[current_gate] = {}
            elif line.startswith('OP:') and current_gate:
                gate_info[current_gate]['operation'] = line.split(':')[1].strip()
            elif line.startswith('COST:') and current_gate:
                gate_info[current_gate]['cost'] = float(line.split(':')[1].strip())
            elif line.startswith('DELAY:') and current_gate:
                delay_values = line.split(':')[1].strip().split()
                gate_info[current_gate]['delay'] = list(map(float, delay_values))

    return gate_info

def print_gate_info(gate_info):
    for gate, info in gate_info.items():
        print(f"Gate: {gate}, Operation: {info['operation']}, Cost: {info['cost']}, Delay: {info['delay']}")

# Example usage

    # Replace 'path_to_file.time' with the path to your actual .time file
#gate_info = parse_gate_time_file('cell_library.time')


# def assign_random_costs(graph):
#     # Identify all unique gate types in the graph
#     gate_types = set(edge[2]['gate'] for edge in graph.edges(data=True))
#     # Assign a random cost to each gate type
#     gate_costs = {gate: random.randint(1, 10) for gate in gate_types}
#     return gate_costs

def calculate_path_costs(graph, paths, gate_info):
    path_costs = {}
    top_cost=0
    for path in paths:
        total_cost = 0
        # Calculate the total cost for the current path
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1])
            gate_type = edge_data['gate']
            total_cost += gate_info[gate_type+'1']['cost']
        #path_costs.append((path, total_cost))
        path_costs[str(path)]=total_cost
        top_cost+=total_cost
    return path_costs, top_cost

def calculate_path_cost(graph, path, gate_info):
    total_cost = 0
    # Calculate the total cost for the current path
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i+1])
        gate_type = edge_data['gate']
        total_cost += gate_info[gate_type+'1']['cost']
    #path_costs.append((path, total_cost))
    return total_cost

def find_all_io_paths(graph):
    input_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'input']
    output_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'output']
    all_paths = []
    for input_node in input_nodes:
        for output_node in output_nodes:
            paths = list(nx.all_simple_paths(graph, source=input_node, target=output_node))
            all_paths.extend(paths)
    return all_paths

def print_path_costs(path_costs):
    for path, cost in path_costs:
        print("Path: {} -> Total Cost: {}".format(" -> ".join(path), cost))

# Example usage

#path_costs, total_cost = calculate_path_costs(graph, paths,gate_info)
#print(total_cost)

# OPERATIONS
def add(a,b):
  g=[]
  g.append(a[0]+b[0])
  g.append(a[1]+b[1])
  g.append(a[2]+b[2])
  g.append(np.sqrt(a[3]**2 + b[3]**2))
  return g

def fi(K9):
  if K9 < 0:
      return  erfc(abs(K9))
  else:
      return 0.5 + 0.5 * erf(K9 / np.sqrt(2))


def max(a,b):
  sigmaA=np.sqrt(a[1]**2 + a[2]**2 + a[3]**2)
  sigmaB=np.sqrt(b[1]**2 + b[2]**2 + b[3]**2)
  rho=(((a[1]*b[1])+(a[2]*b[2]))/(sigmaA*sigmaB))
  theta=np.sqrt(sigmaA**2+sigmaB**2 -2*rho*sigmaA*sigmaB)
  #print(theta)
  a0b0t=(a[0]-b[0])/theta
  phi=fi(a0b0t)
  phi2=(1/np.sqrt(2*3.1415926))*np.exp(-(a0b0t**2)/2)
  g0=a[0]*phi+b[0]*(1-phi)+theta*phi2
  g1=a[1]*phi+b[1]*(1-phi)
  g2=a[2]*phi+b[2]*(1-phi)
  varmax=(sigmaA**2 + a[0]**2)*phi + (sigmaB**2+b[0]**2)*(1-phi) + (a[0]+b[0])*theta*phi2 - g0**2
  g3=np.sqrt(abs(varmax-g1**2 - g2**2))
  return [g0,g1,g2,g3]

def max2(a,b):
  sigmaA=np.sqrt(a[1]**2 + a[2]**2 + a[3]**2)
  sigmaB=np.sqrt(b[1]**2 + b[2]**2 + b[3]**2)
  rho=(((a[1]*b[1])+(a[2]*b[2]))/(sigmaA*sigmaB))
  theta=np.sqrt(sigmaA**2+sigmaB**2 -2*rho*sigmaA*sigmaB)
  #print(theta)
  a0b0t=(a[0]-b[0])/theta
  phi=fi(a0b0t)
  return phi

def max3(a,b):
  if a[0]>b[0]:
    return 0
  elif a[0]==b[0]:
    if a[1]>b[1]:
      return 0
    elif a[1]==b[1]:
      if a[2]>b[2]:
        return 0
      else: 
        return 1
    else:
      return 1
  else:
    return 1


def min(a,b):
  sigmaA=np.sqrt(a[1]**2 + a[2]**2 + a[3]**2)
  sigmaB=np.sqrt(b[1]**2 + b[2]**2 + b[3]**2)
  rho=(((a[1]*b[1])+(a[2]*b[2]))/(sigmaA*sigmaB))
  theta=np.sqrt(sigmaA**2+sigmaB**2 -2*rho*sigmaA*sigmaB)
  print(theta)
  a0b0t=(a[0]-b[0])/theta
  phi=fi(a0b0t)
  phi2=(1/np.sqrt(2*3.1415926))*np.exp(-(a0b0t**2)/2)
  g0=a[0]*(1-phi)+b[0]*(phi)-theta*phi2
  g1=b[1]*phi+a[1]*(1-phi)
  g2=b[2]*phi+a[2]*(1-phi)
  varmin=(sigmaA**2 + a[0]**2)*(1-phi) + (sigmaB**2+b[0]**2)*phi - (a[0]+b[0])*theta*phi2 - g0**2
  g3=np.sqrt(varmin-g1**2 - g2**2)
  return [g0,g1,g2,g3]

import networkx as nx

def ssta(graph):
    # Initialize delays for each node
    node_delays = {node: [0, 0, 0, 0] for node in graph.nodes()}
    prev_dict={}
    # Breadth-first search to traverse the graph
    for node in nx.topological_sort(graph):
      #skip if the node is level 1
      if list(graph.predecessors(node)) == [] :
        continue

      #compute all the path delays up to that node
      delays=[]
      for pred in graph.predecessors(node):
        edge_delay= interconnect_timings[(pred,node)]
        gate_delay = gate_info[get_gate_info(graph, pred, node)+'1']['delay']
        delay= add(node_delays[pred],add(edge_delay,gate_delay))
        delays.append(delay)

      max_id=0
      for i in range(1,len(delays)-1):
        if max2(delays[i],delays[i+1]) >= 0.5:
          max_id=i
        else:
          max_id=i+1

      prev_dict[node]=list(graph.predecessors(node))[max_id]
      #print(prev_dict)




      max_delay=delays[0]
      #max all the paths up to that node
      for i in range(1,len(delays)):
        max_delay=max(max_delay,delays[i])
      node_delays[node]=max_delay

    return node_delays,prev_dict

def is_input_node(graph, node):
    if node in graph:
        # Get node data and check if the type is 'input'
        node_data = graph.nodes[node]
        return node_data.get('type') == 'input'
    else:
        return False  # Node does not exist in the graph

def find_critical_path(graph,prev_dict):
  output_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'output']
  critical_paths=[]
  latest_out=output_nodes[0]
  for i in range(0,len(output_nodes)-1):
    if max2(node_delays[output_nodes[i]],node_delays[output_nodes[i+1]]) >= 0.5:
      latest_out=output_nodes[i]
    else:
      latest_out=output_nodes[i+1]
      
  output=latest_out
  critical_path=[output]
  node=output
  while not is_input_node(graph,node):
    node=prev_dict[node]
    critical_path.append(node)

  return list(reversed(critical_path))

def print_path(path):
    print(" -> ".join(path))

# Assuming `graph` is already created and `gate_delays` is obtained from parse_gate_time_file function
# Example usage:

# Load the graph and gate delays, example initialization # this should be your actual graph initialization  # this should be populated by your gate parsing function

    # Populate the graph and gate_delays dictionary as required

# Perform static timing analysis

#node_delays,prev_dict=ssta(graph)

#find critical path
#critical_path=find_critical_path(graph,prev_dict)
#print("Critical path: "," -> ".join(critical_path))
#print("Critical path cost: ", path_costs[str(critical_path)])

def ssta_opt(graph,changed_gates):
    # Initialize delays for each node
    node_delays = {node: [0, 0, 0, 0] for node in graph.nodes()}
    prev_dict={}
    # Breadth-first search to traverse the graph
    change=1
    for node in nx.topological_sort(graph):
      #skip if the node is level 1
      if list(graph.predecessors(node)) == [] :
        continue

      #compute all the path delays up to that node
      delays=[]
      for pred in graph.predecessors(node):
        type='1'
        if (random.randint(1,100)<30 and change) or ((pred,node) in changed_gates):
          type='3'
          #change_made.append(pred + "->" + node + ": 1 to 2")
          changed_gates.append((pred,node))
          if (pred,node) not in changed_gates:
            change=0
        edge_delay= interconnect_timings[(pred,node)]
        gate_delay = gate_info[get_gate_info(graph, pred, node)+type]['delay']
        delay= add(node_delays[pred],add(edge_delay,gate_delay))
        delays.append(delay)

      max_id=0
      for i in range(1,len(delays)-1):
        if max2(delays[i],delays[i+1]) >= 0.5:
          max_id=i
        else:
          max_id=i+1

      prev_dict[node]=list(graph.predecessors(node))[max_id]
      #print(prev_dict)
      
      max_delay=delays[0]
      #max all the paths up to that node
      for i in range(1,len(delays)):
        max_delay=max(max_delay,delays[i])
      node_delays[node]=max_delay

    return node_delays,prev_dict,changed_gates


def modified_path_cost(graph, path, gate_info, changed_gates):
    total_cost = 0
    # Calculate the total cost for the current path
    for i in range(len(path) - 1):
        type='1'
        if (path[i], path[i+1]) in changed_gates:
          type='3'
        edge_data = graph.get_edge_data(path[i], path[i+1])
        gate_type = edge_data['gate']
        total_cost += gate_info[gate_type+type]['cost']
    #path_costs.append((path, total_cost))
    return total_cost

def modified_path_costs(graph, paths, gate_info,changed_gates):
    path_costs = {}
    top_cost=0
    for path in paths:
        total_cost = 0
        # Calculate the total cost for the current path
        for i in range(len(path) - 1):
            type='1'
            if (path[i], path[i+1]) in changed_gates:
              type='3'
            edge_data = graph.get_edge_data(path[i], path[i+1])
            gate_type = edge_data['gate']
            total_cost += gate_info[gate_type+type]['cost']
        #path_costs.append((path, total_cost))
        path_costs[str(path)]=total_cost
        top_cost+=total_cost
    return path_costs, top_cost


circuit='c1908'

#parse the circuit into a graph 
graph = parse_bench_file(circuit+'.bench')
#paths = find_all_io_paths(graph)

#read the interconnect delays of the circuit from file
interconnect_timings = parse_time_file(circuit+'.time')

#read gate delays from file
gate_info = parse_gate_time_file('cell_library.time')

#calculate costs for each path
#path_costs, total_cost = calculate_path_costs(graph, paths,gate_info)

#run ssta
node_delays,prev_dict=ssta(graph)

#find critical path
critical_path=find_critical_path(graph,prev_dict)
print("Critical path: "," -> ".join(critical_path))
print("Critical path cost: ", calculate_path_cost(graph, critical_path, gate_info))
#print("Total cost of the circuit: ",total_cost)

print(node_delays[critical_path[-1]])


