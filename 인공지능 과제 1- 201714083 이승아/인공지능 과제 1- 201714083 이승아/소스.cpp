#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <random>
#include <time.h>
#include <Windows.h>
#include <cmath>

#include <climits>
using namespace std;

static const int INPUT_NODE_NUM = 21;
static const int HIDDEN_NODE_NUM = 2000;
static const double learningRate = 0.0005;

void main()
{
	random_device rd2;		mt19937_64 VALUE(rd2());
	uniform_real_distribution<> RANDOM(0, 4.8);

	int epoch = 1;

	// Data input
	ifstream infile("CTG_data_input.csv");
	vector<vector<double>> All_Data;
	if (infile.good())
	{
		string line;	int cnt = 0;
		while (getline(infile, line))
		{
			if (cnt > 2126)
				break;
			if (cnt == 0)
			{
				++cnt;
				continue;
			}
			stringstream sep(line);
			string field;
			vector<double>line_data;
			while (getline(sep, field, ','))
				line_data.push_back(stod(field));
			All_Data.push_back(line_data);
			++cnt;
		}
	}

	// 8 : 2�� ������ �ֱ�.
	int full_size = All_Data.size();
	int train_size = full_size*0.8;	//1700	
	int test_size = full_size - train_size;	// 425
	int train_count = train_size; int test_count = test_size;

	vector<vector<double>> Train_datas;
	vector<vector<double>> Test_datas;

	bool *Check_Picked = new bool[full_size];

	for (int i = 0; i < full_size; i++)
		Check_Picked[i] = false;

	random_device rd;		mt19937_64 INDEX(rd());
	uniform_real_distribution<> PICK_INDEX_RANDOMLY(0, full_size - 1);
	int count = 0;

	// random�ϰ� Training Set�� Test Set �з�.
	while (train_count != 0)
	{
		int index = PICK_INDEX_RANDOMLY(INDEX);
		if (Check_Picked[index] == false)
		{
			Check_Picked[index] = true;
			vector<double> line_data;
			for (int i = 0; i < 22; i++)
				line_data.push_back(All_Data[index][i]);
			Train_datas.push_back(line_data);
			train_count--;
		}
	}

	count = 0;

	for (int i = 0; i < full_size; i++)
	{
		if (Check_Picked[i] == false)
		{
			vector<double> line_data;
			for (int j = 0; j < 22; j++)	
				line_data.push_back(All_Data[i][j]);
			Test_datas.push_back(line_data);
			test_count--;
		}
	}

	double input_Arr[INPUT_NODE_NUM] = { 0, };
	double hiddenWeightArr[HIDDEN_NODE_NUM][INPUT_NODE_NUM] = { 0, };
	double hiddenResultArr[HIDDEN_NODE_NUM] = { 0, };
	double hiddenCriticalArr[HIDDEN_NODE_NUM] = { 0, };
	double hiddenErrorArr[HIDDEN_NODE_NUM] = { 0, };
	double outputAnswerArr[3] = { 0, };
	double outputResultArr[3] = { 0, };
	double outputWeightArr[3][HIDDEN_NODE_NUM] = { 0, };
	double outputCriticalArr[3] = { 0, };
	double outputErrorArr[3] = { 0, };

	for (int i = 0; i < HIDDEN_NODE_NUM; i++)
	{
		for (int j = 0; j < INPUT_NODE_NUM; j++)
		{
			hiddenWeightArr[i][j] = (RANDOM(VALUE) - 2.4) / INPUT_NODE_NUM;	// ������ ����ġ �ʱ�ȭ.
		}
		hiddenCriticalArr[i] = (RANDOM(VALUE) - 2.4) / INPUT_NODE_NUM;		// ������ �Ӱ谪 ����.
	}
	for (int k = 0; k < 3; k++)
	{
		for (int j = 0; j < HIDDEN_NODE_NUM; j++)
		{
			outputWeightArr[k][j] = (RANDOM(VALUE) - 2.4) / INPUT_NODE_NUM;	// ����� ����ġ �ʱ�ȭ.
		}
		outputCriticalArr[k] = (RANDOM(VALUE) - 2.4) / INPUT_NODE_NUM;		// ����� �Ӱ谪 ����.
	}

	int end = 500;

	for (int a = 1; a <= end; a++)
	{
		int checkLearningData[3] = { 0, };		int checkTestingData[3] = { 0, };
		int rightLearningData[3] = { 0, };		int rightTestingData[3] = { 0, };
		int percent = 0;		int count = 0;		int n;
		double errorTestingData = 0;	double targetFunc_TrainingResult = 0;	double targetFunc_TestResult = 0;

		while (train_count != train_size)
		{
			//�Է���
			for (int i = 0; i < 21; i++)
			{
				input_Arr[i] = Train_datas[train_count][i];
			}
			n = (int)((Train_datas[train_count][21]) - 1);
			checkLearningData[n]++;

			// ���� ���� �� -> ����� ���䰪 �迭 ����.
			for (int i = 0; i < 3; i++)
			{
				if (i == n)
					outputAnswerArr[i] = 1;
				else
					outputAnswerArr[i] = 0;
			}

			/* FeedForward / Ȱ��ȭ
			������ ����ġ > Ȱ��ȭ �Լ�(Sigmoid �Լ�) ��� > ������ ��°� �迭 ����.
			������ ��°� �迭 > Ȱ��ȭ �Լ�(Sigmoid �Լ�) > ����� ��� �迭 ����.
			����� ���� ���� ��� > ����ġ ������ ��� > ��� ����ġ ����.
			*/

			// ������ ���� ��� ���.
			for (int j = 0; j < HIDDEN_NODE_NUM; j++)
			{
				double result = 0;
				double sum = 0;
				for (int i = 0; i < INPUT_NODE_NUM; i++)
				{
					sum += hiddenWeightArr[j][i] * input_Arr[i];
				}
				result = 1.0 / (1 + exp(-sum));	// Ȱ��ȭ �Լ� sigmoid
				hiddenResultArr[j] = result;
			}

			// ����� ���� ��� ���.
			for (int k = 0; k < 3; k++)
			{
				double result = 0;
				double sum = 0;
				for (int j = 0; j < HIDDEN_NODE_NUM; j++)
				{
					sum += hiddenResultArr[j] * outputWeightArr[k][j];
				}
				result = 1.0 / (1 + exp(-sum));
				outputResultArr[k] = result;
			}


			/* Backpropagation / ��� ������ ������ ������ ���������� ���Ľ�Ű�� ������ �Ű���� ����ġ�� ����.
			������� �ִ� ������ ���� ���� ���⸦ ���, ������ ���, ��� ���������� ����ġ ����.
			*/

			for (int k = 0; k < 3; k++)
			{
				outputErrorArr[k] = outputResultArr[k] * (1 - outputResultArr[k])*(outputAnswerArr[k] - outputResultArr[k]);
				for (int j = 0; j < HIDDEN_NODE_NUM; j++)
				{
					outputWeightArr[k][j] += learningRate*hiddenResultArr[j] * outputErrorArr[k];
				}
			}
			//������ ���� ���� ���
			for (int j = 0; j < HIDDEN_NODE_NUM; j++)
			{
				double sum = 0;
				for (int k = 0; k < 3; k++)
					sum += outputErrorArr[k] * outputWeightArr[k][j];
				hiddenErrorArr[j] = hiddenResultArr[j] * (1 - hiddenResultArr[j])*sum;
			}
			//������ ����ġ ����
			for (int j = 0; j < HIDDEN_NODE_NUM; j++)
			{
				for (int i = 0; i < INPUT_NODE_NUM; i++)
				{
					hiddenWeightArr[j][i] += learningRate*input_Arr[i] * hiddenErrorArr[j];
				}
			}

			// ������� ���� ū �� ã��.
			int maxIndex = -1; double max = 0;
			for (int k = 0; k < 3; k++)
			{
				if (max < outputResultArr[k])
				{
					max = outputResultArr[k];
					maxIndex = k;
				}
			}
			if (n == maxIndex)
			{
				percent++;
				rightLearningData[n]++;
			}

			for (int t = 0; t < 3; t++)
			{
				if (t == n)
					errorTestingData += (1 - outputResultArr[t])*(1 - outputResultArr[t]);
				else
					errorTestingData += outputResultArr[t] * outputResultArr[t];
			}

			// TargetFunction
			for (int i = 0; i < 3; i++)
			{
				targetFunc_TrainingResult += pow(outputAnswerArr[i] - outputResultArr[i], 2);
			}
			train_count++;
		}

		std::cout << "Train's targetFunction Result : " << targetFunc_TrainingResult / 2 << endl;
		std::cout << "Training ���� epoch " << epoch << "-> Accuracy : " << (double)percent * 100 / train_size << "% , Loss : " << errorTestingData / train_size << endl;

		percent = 0;	count = 0;

		/* Test Data �� Training Data�� ���� ������Ʈ�� ����ġ�� �̿��Ͽ�
		������� �̾Ƴ� ���䰪�� ���ϴ� �͸� �����Ѵ�.(FeedForward�� ����.)
		*/
		while (test_count != test_size)
		{
			//�Է�
			for (int i = 0; i < 21; i++)
			{
				input_Arr[i] = Test_datas[test_count][i];
			}
			n = (int)((Test_datas[test_count][21]) - 1);
			checkLearningData[n]++;

			// ������ ���� ��� ���.
			for (int j = 0; j < HIDDEN_NODE_NUM; j++)
			{
				double result = 0;
				double sum = 0;
				for (int i = 0; i < INPUT_NODE_NUM; i++)
				{
					sum += hiddenWeightArr[j][i] * input_Arr[i];
				}
				sum -= hiddenCriticalArr[j];	// Test Data������ �Ӱ谪 ���.
				result = 1.0 / (1 + exp(-sum));
				hiddenResultArr[j] = result;
			}
			// ����� ���� ��� ���.
			for (int k = 0; k < 3; k++)
			{
				double result = 0;
				double sum = 0;
				for (int j = 0; j < HIDDEN_NODE_NUM; j++)
				{
					sum += hiddenResultArr[j] * outputWeightArr[k][j];
				}
				sum -= outputCriticalArr[k];
				result = 1.0 / (1 + exp(-sum));
				outputResultArr[k] = result;
			}

			int maxIndex = -1; double max = 0;
			for (int k = 0; k < 3; k++)
			{
				if (max < outputResultArr[k])
				{
					max = outputResultArr[k];
					maxIndex = k;
				}
			}
			if (n == maxIndex)
			{
				rightTestingData[n]++;
				percent++;
			}
			count++;
			for (int t = 0; t < 3; t++)
			{
				if (t == n)
					errorTestingData += (1 - outputResultArr[t])*(1 - outputResultArr[t]);
				else
					errorTestingData += outputResultArr[t] * outputResultArr[t];
			}

			test_count++;

			if (a == end)
			{
				for (int i = 0; i < 3; i++)
				{
					targetFunc_TestResult += pow(outputAnswerArr[i] - outputResultArr[i], 2);
				}
			}
		}
		if (a == end)
			std::cout << "Test's targetFunction Result : " << targetFunc_TestResult / 2 << endl;
		std::cout << "Test ���� epoch " << epoch << "-> Accuracy : " << (double)percent * 100 / test_size << "% , Loss : " << errorTestingData / test_size << endl;

		epoch++;	train_count = 0; test_count = 0;
		std::cout << "---------------------------------------------------------------------" << endl;

	}

}
