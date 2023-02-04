#include"priorityQueue.h"
NodeM priorityQueue::deQueue() {

	NodeM data;

	data = arrayN[1];
	arrayN[1] = arrayN[currentSize--];

	percolateDown(1);
	return data;
}

void priorityQueue::percolateDown(int hole) {
	int child;

	NodeM tmp = arrayN[hole];

	for (; hole * 2 <= currentSize; hole = child) {
		child = hole * 2;
		if (child != currentSize && arrayN[child + 1].totalNum < arrayN[child].totalNum)
			child++;
		if (arrayN[child].totalNum < tmp.totalNum) arrayN[hole] = arrayN[child];
		else break;
	}
	arrayN[hole] = tmp;
}


void priorityQueue::buildHeap() {
	for (int i = currentSize / 2; i > 0; i--) {
		percolateDown(i);
	}
}


priorityQueue::priorityQueue(const Nodem data, int size) :maxSize(size + 10), currentSize(size) {

	arrayN = new NodeM[maxSize];

	for (int i = 0; i < size; i++) arrayN[i + 1] = data[i];
	buildHeap();
}


void priorityQueue::doubleSpace() {
	Nodem tmp = arrayN;

	maxSize *= 2;
	arrayN = new NodeM[maxSize];
	for (int i = 0; i <= currentSize; ++i) arrayN[i] = tmp[i];
	delete[] tmp;
}

void priorityQueue::enQueue(NodeM data) {

	if (currentSize == maxSize - 1) doubleSpace();
	//ÏòÉÏ¹ýÂË
	int hole = ++currentSize;

	for (; hole > 1 && data.totalNum < arrayN[hole / 2].totalNum; hole /= 2) {
		arrayN[hole] = arrayN[hole / 2];
	}
	arrayN[hole] = data;


}

NodeM priorityQueue::getHead() const {

	return arrayN[1];
}

int priorityQueue::getSize() {
	return currentSize;
}
priorityQueue::priorityQueue() {

	int capacity = 7000;

	arrayN = new NodeM[capacity];
	maxSize = capacity;
	currentSize = 0;
}

//priorityQueue(const Type data[], int size);

priorityQueue::~priorityQueue() {

	delete[] arrayN;

}