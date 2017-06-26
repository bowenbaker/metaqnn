import json

# Define messages to be sent between server and clients

def parse_message(msg):
    '''takes message with format PROTOCOL and returns a dictionary'''
    return json.loads(msg)

def construct_login_message(hostname):
    return json.dumps({'sender': hostname,
                       'type': 'login'})

def construct_new_net_message(hostname, net_string, epsilon, iteration_number):
    return json.dumps({'sender': hostname,
                       'type': 'new_net',
                       'net_string': net_string,
                       'epsilon': epsilon,
                       'iteration_number': iteration_number})

def construct_net_trained_message(hostname,
                                  net_string,
                                  acc_best_val,
                                  iter_best_val,
                                  acc_last_val,
                                  iter_last_val,
                                  epsilon,
                                  iteration_number):
    return json.dumps({'sender': hostname,
                       'type': 'net_trained',
                       'net_string': net_string,
                       'acc_best_val': acc_best_val,
                       'iter_best_val': iter_best_val,
                       'acc_last_val': acc_last_val,
                       'iter_last_val': iter_last_val,
                       'epsilon': epsilon,
                       'iteration_number': iteration_number})

def construct_net_too_large_message(hostname):
    return json.dumps({'sender': hostname,
                       'type': 'net_too_large'})

def construct_redundant_connection_message(hostname):
    return json.dumps({'sender': hostname,
                       'type': 'redundant_connection'})
            