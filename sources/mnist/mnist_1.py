import numpy
import matploylib as plt

class NeuralNetwork:
    # 초기화
    def __init__(self, inputnodes, hiddennodes, outputnodes, learnrate):
        pass

    # 학습
    def train(self, inputs_list, targets_list):
        pass

    # 검증(테스트)
    def test(self, inputs_list):
        pass

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 학습 데이터(샘플 100) 파일 읽어오기
training_data_file = open('../dataset/mnist_train_100.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    # record의 값을 [,]로 구분
    all_valeus = record.split(',')
    # 값 조정 -> X/255.0 * 0.99 + 0.01
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    # target 리스트 준비 (10개짜리 빈 list)
    targets = numpy.zeros(output_nodes) + 0.01
    # 정답을 ont hot encoding으로 표시 
    targets[int(all_values[0])] = 0.99
    model.train(inputs, targets) 
    pass

'''
model 검증
'''
# 테스트 데이터 읽어오기
test_data_file = open('../dataset/mnist_test_10.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[9].split(',')
print(all_valeus[0])
print(model.test((numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01))
