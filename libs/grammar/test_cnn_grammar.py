from yapps import cli_tool
cli_tool.generate('cnn.g')
import cnn
print cnn.parse('net', '[C(1,2,3), C(1,2,3), SM(10)]')
print cnn.parse('net', '[C(1,2,3), C(4, 5, 6), FC(40), FC(40), SM(10)]')
print cnn.parse('net', '[C(1,1,1), S{[C(2,2,2), C(22,22,22)], [C(3,3,3)]}, C(4,4,4), P(2), SM(10)]')
print cnn.parse('net', '[C(1,2,3), NIN(100), BN, NIN(100), GAP(100)]')
