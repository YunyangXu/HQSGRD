#pragma once


#ifndef PRIORITYQUEUE_PRIORITYQUEUE_H
#define PRIORITYQUEUE_PRIORITYQUEUE_H

typedef struct NodeM {
	int totalNum = 0;
	int CalculateSign;
	int sign;

	int old;

	float TotalL = 0;
	float TotalA = 0;
	float TotalB = 0;

	int PixelGrayMax;
	int PixelGrayMin;

	float StandardDeviation = 0;

	float StandardDeviationR = 0;
	float StandardDeviationG = 0;
	float StandardDeviationB = 0;

	int initial_x;//初始位置的X
	int initial_y;//初始位置的Y
	int AX;
	int AY;
}*Nodem;


class priorityQueue {
public:

	priorityQueue();

	//priorityQueue(const Type data[], int size);

	~priorityQueue();
	bool isEmpty() const { return currentSize == 0; }

	//void enQueue(const Type &x);

	void enQueue( NodeM num);

	NodeM getHead() const;

	priorityQueue(const Nodem data, int size);

	NodeM deQueue();

	int getSize();

private:
	int currentSize;
	Nodem arrayN;
	int maxSize;
	void doubleSpace();
	void buildHeap();
	void percolateDown(int hole);
};



#endif //PRIORITYQUEUE_PRIORITYQUEUE_H
