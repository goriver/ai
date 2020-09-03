import numpy
import matplotlib as plt
import scipy.special

class NeuralNetwork:
    # 초기화
    def __init__(self, inputnodes, hiddennodes, outputnodes, learnrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 가중치 행렬 wih, who 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)) 
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learnrate

        # 활성화 함수 지정 
        #self.activation_func = lambda x : scipy.special.expit(x)
        # activation_func(5)
        # sigmoid(5)
        pass

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))
    
    # 학습
    def train(self, inputs_list, targets_list):
        # 1.입력 리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 2. 은닉계층으로 들어오는 값 계산 (input -> hidden) 
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 3. 은닉계층에서 나가는 값 계산 (hidden -> output) -> sigmoid
        hidden_outputs = self.sigmoid(hidden_inputs)

        # 4. 최종계층으로 들어오는 값 계산 (hidden -> output)
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 5. 최종계층에서 나가는 값 계산 
        final_outputs = self.sigmoid(final_inputs)

        # 6. 오차 계산 (실제 값 - 모델에 의해서 계산 된 값)
        output_errors = targets - final_outputs

        # 7. 오차역전법 (은닉계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들의 재조합)
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 8. 은닉계층과 출력계층 간 가중치 업데이트
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1- final_outputs)), numpy.transpose(hidden_outputs))

        # 9. 입력계층과 은닉계층 간 가중치 업데이트 
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1- hidden_outputs)), numpy.transpose(inputs))

        pass

    # 검증(테스트)
    def test(self, inputs_list):
        # 1. 입력 리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 2. 은닉계층으로 들어오는 값 계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 3. 은닉계층으로 나가는 값 계산
        hidden_outputs = self.sigmoid(hidden_inputs)
        # 4. 최종계층으로 들어오는 값 계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 5. 최종계층으로 나가는 값 계산 
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 학습 데이터(샘플 100) 파일 읽어오기
training_data_file = open('../dataset/mnist_train.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 10

for e in range(epochs):
    for record in training_data_list:
        # record의 값을 [,]로 구분
        all_values = record.split(',')
        # 값 조정 -> X/255.0 * 0.99 + 0.01
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
        # target 리스트 준비 (10개짜리 빈 list)
        targets = numpy.zeros(output_nodes) + 0.01
        # 정답을 ont hot encoding으로 표시 
        targets[int(all_values[0])] = 0.99
        model.train(inputs, targets) 
        pass
    pass

'''
model 검증
'''
# 테스트 데이터 읽어오기
test_data_file = open('../dataset/mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# all_values = test_data_list[2].split(',')
# print(all_values[0])
# predict = model.test((numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01)
# print(predict)
# print(numpy.argmax(predict))

scorecard = []
for record in test_data_list:
    all_values = record.split(',') # 0 -> label, 1 ~ 입력값 
    correct_label = int(all_values[0])
    print('correct label = ', correct_label)
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = model.test(inputs)
    predict = numpy.argmax(outputs)
    print('model predict = ', predict)
    if (predict == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print("accuracy_score=", scorecard_array.sum() / scorecard_array.size)