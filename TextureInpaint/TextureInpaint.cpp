#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<set>
#include<hash_set>
#include<algorithm>
#include "Eigen/Eigen"

#include "nanoflann.hpp"
#include "cloudpoints.h"

#include "MRFEnergy.h"

#include <codex_utils.h>
//-------------Parameters
const int M = 3;
const int R = 1000;
const int nPointsMax = 70000;
const int nIteration = 1 << M;
const int kapa = 5;
const int nColorDim=7;
const double miuX = 0.05;
const double niuP = 30;
const double zeta = 0.5;
const double beta = 0.6;
//-----------------------
typedef nanoflann::KDTreeSingleIndexAdaptor <
	nanoflann::L2_Simple_Adaptor<double, PointCloud<double> >,
	PointCloud<double>,
	3
> KDTree;
KDTree   *pointKDtree,*lcKDtree=NULL;
//-----------------------
int pCandidates[nIteration][nPointsMax];
int deltaCandidates[nIteration][nPointsMax];
Eigen::Vector3d xPoints[nPointsMax];
PointCloud<double> xPointsCloud,lcPointsCloud;

Eigen::Matrix<double,nColorDim,1> pTexture[nPointsMax],pResult[nPointsMax];

Eigen::Vector3d nResult[nPointsMax];
std::set<int> lc, ll, lm, lv;
//------------------------
int tlc[nPointsMax], tll[nPointsMax], tlm[nPointsMax];
int nPoints = 0;
//------------------------
codex::utils::timer time_counter;
//------------------------
void export_pointcloud_ply(const char *filename, Eigen::Vector3d *pos, int n,
	const Eigen::Vector3d *p_normal,
	const Eigen::Matrix<double, nColorDim, 1> *p_color);

void initializeCandidates(){
	for (int i = 0; i < nIteration; i++)
		for (int j = 0; j < nPoints; j++)
		{
			pCandidates[i][j] = -1;
			deltaCandidates[i][j] = -1;
		}
}
void generateKDTree(){
	xPointsCloud.pts.resize(nPoints);
	for (size_t i = 0; i < nPoints; i++)
	{
		xPointsCloud.pts[i].x = xPoints[i](0);
		xPointsCloud.pts[i].y = xPoints[i](1);
		xPointsCloud.pts[i].z = xPoints[i](2);
	}
	pointKDtree = new KDTree(3, xPointsCloud, nanoflann::KDTreeSingleIndexAdaptorParams(20));
	pointKDtree->buildIndex();	
}
void generateCube(){
	nPoints = 60000;
	for (int i = 0; i < nPoints; i++)
	{
		int faceIndex = i / (nPoints/6);
		double x = rand()*2.0 / RAND_MAX-1.0;
		double y = rand()*2.0 / RAND_MAX-1.0;
		//x = y = 0;
		switch (faceIndex){
			case 0: xPoints[i] << +1.0, x, y; break;
			case 1: xPoints[i] << x, -1.0, y; break;
			case 2: xPoints[i] << -1.0, x, y; break;
			case 3: xPoints[i] << x, +1.0, y; break;
			case 4: xPoints[i] << x, y, +1.0; break;
			case 5: xPoints[i] << x, y, -1.0; break;
		}
		pTexture[i] << rand()*1.0 / RAND_MAX, rand()*1.0 / RAND_MAX, rand()*1.0 / RAND_MAX;
	}
	lc.clear();
	ll.clear();
	initializeCandidates();
	for (int i = 0; i < (nPoints / 6); i++)
	{
		lc.insert(i);
		pCandidates[0][i] = i;
	}
	for (int i = (nPoints / 6); i < nPoints; i++)
		ll.insert(i);
	lv = lc;
	for (int i = 0; i < (nPoints / 6); i++)
	{
		int dim=4;
		double x = xPoints[i](1), y = xPoints[i](2);
		int nx, ny;
		x = (x + 1) / 2;
		y = (y + 1) / 2;
		nx = x / (1.0 / dim);
		ny = y / (1.0 / dim);
		if (0 == (nx + ny) % 2)
			pTexture[i] << 1.0, 0.0, 0.0;
		else
			pTexture[i] << 1.0, 1.0, 1.0;
	}
}
void readPly(){
	FILE *f;

	fopen_s(&f, "pts_pure.ply", "r");

	fscanf_s(f, "%d", &nPoints);
	for (int i = 0; i < nPoints; i++){
		double x, y, z;
		fscanf_s(f, "%lf %lf %lf", &x, &y, &z);
		xPoints[i] << x, y, z;
	}
	fclose(f);
	int pN;

	fopen_s(&f, "attr.txt", "r");
	fscanf_s(f, "%d", &pN);
	assert(pN == nPoints);
	lc.clear();
	ll.clear();
	initializeCandidates();
	double tmpTexture[nColorDim];
	for (int i = 0; i < nPoints; i++){
		for (int j = 0; j < nColorDim; j++){
			fscanf_s(f, "%lf", &tmpTexture[j]);
			pTexture[i](j) = tmpTexture[j];
		}
		if (tmpTexture[0] < 0)
		{
			ll.insert(i);
		}
		else
		{
			lc.insert(i);
			pCandidates[0][i] = i;
		}
	}
	fclose(f);
	lv = lc;

	//std::cout << "8\n" << pTexture[8] << "\n" << "11\n" << pTexture[11] << "\n";
	//system("pause");

}
void generatelcKDTree(){
	lcPointsCloud.pts.resize(lc.size());
	int len = lc.size();
	for (size_t i = 0; i < len; i++)
	{
		lcPointsCloud.pts[i].x = xPoints[tlc[i]](0);
		lcPointsCloud.pts[i].y = xPoints[tlc[i]](1);
		lcPointsCloud.pts[i].z = xPoints[tlc[i]](2);
	}
	if (lcKDtree) delete lcKDtree;
	lcKDtree = new KDTree(3, lcPointsCloud, nanoflann::KDTreeSingleIndexAdaptorParams(30));
	lcKDtree->buildIndex();
}
double disTmp[nPointsMax];
void computeSymmetry(){
	int nM = 0;
	int lrCount;
	double dM = 0, dR;
	Eigen::Vector3d normalM;
	Eigen::Vector3d normalR, dnormalRd, dnormalMd,tmp;
	Eigen::Matrix3d ImNormalR2, ImNormalM2;
	for (int m = 0; m < M; m++){
		printf("Layer Transfer %d iteration\n", m+1);
		int inxll = 0;
		for (std::set<int>::iterator it = ll.begin(); it != ll.end(); it++, inxll++)
			tll[inxll] = *it;
		int inxlc = 0;
		for (std::set<int>::iterator it = lc.begin(); it != lc.end(); it++, inxlc++)
			tlc[inxlc] = *it;
		generatelcKDTree();
		nM = 0;
		dM = 0.0;
		normalM << 0, 0, 0;
		lm.clear();
		int lastPro = 0;
		const size_t numResults = 1;
		size_t retIndex[numResults];
		double outDistSqr[numResults];
		nanoflann::KNNResultSet<double> resultSet(numResults);
		
		for (int r = 0; r < R; r++){
			lrCount = 0;
			int sr = rand()*rand() % nPoints;
			int tr = rand()*rand() % ll.size();
			tmp = xPoints[sr] - xPoints[tll[tr]];
			normalR = tmp / tmp.norm();
			dR = 0.50*(normalR.transpose()*(xPoints[sr] + xPoints[tll[tr]]))(0);
			ImNormalR2 = Eigen::Matrix3d::Identity()-2.0*normalR*normalR.transpose();
			dnormalRd = 2.0*normalR*dR;
			
			//time_counter.update();
			for (int s = 0; s < inxll; s++){
				
				resultSet.init(&retIndex[0], &outDistSqr[0]);
				Eigen::Vector3d oppoPoint = (ImNormalR2*xPoints[tll[s]] + dnormalRd);
				double queryPt[3] = { oppoPoint(0), oppoPoint(1), oppoPoint(2) };
				lcKDtree->findNeighbors(resultSet, &queryPt[0], nanoflann::SearchParams(10));
				int oppoIndex = tlc[retIndex[0]];
				assert(oppoIndex >= 0 && oppoIndex<nPoints);
				double xVal = (oppoPoint - xPoints[oppoIndex]).norm();
				double pVal = (pTexture[tll[s]] - pTexture[oppoIndex]).norm();
				if (xVal < miuX && pVal < niuP)
					lrCount++;
			
				/*	int flag = -1;
#pragma omp parallel for
				for (int t = 0; t < inxlc; t++){
					if (-1 == flag)
					{
						double xVal = (xPoints[tll[s]] - (ImNormalR2*xPoints[tlc[t]] + dnormalRd)).norm();
						double pVal = (pTexture[tll[s]] - pTexture[tlc[t]]).norm();
						if (xVal < miuX && pVal < niuP)
						{
								flag = 0;			
							//break;
						}
					}
				}
				if (0 == flag)
					lr.insert(tll[s]);*/



			}
			if (lrCount > nM){
				nM = lrCount;
				normalM = normalR;
				dM = dR;
			}
			if (int(r*100.0 / R) > lastPro)
			{
				lastPro = int(r*100.0 / R);
				printf("Ransack %d %%\n", lastPro);
			}
			/*time_counter.update();
			printf("\nTime: %lf\n", time_counter.elapsed_time());
			system("pause");*/
		}
		int iLim = 1 << (m);
		ImNormalM2 = Eigen::Matrix3d::Identity() - 2.0*normalM*normalM.transpose();
		dnormalMd = 2.0*normalM*dM;
		std::cout << normalM << "\n";
		std::cout << dM << "\n";
		std::cout << nM << "\n";
		lastPro = 0;
		lm.clear();
		for (int s = 0; s < inxll; s++){
			resultSet.init(&retIndex[0], &outDistSqr[0]);
			Eigen::Vector3d oppoPoint = (ImNormalM2*xPoints[tll[s]] + dnormalMd);
			double queryPt[3] = { oppoPoint(0), oppoPoint(1), oppoPoint(2) };
			lcKDtree->findNeighbors(resultSet, &queryPt[0], nanoflann::SearchParams(10));
			int oppoIndex = tlc[retIndex[0]];
			assert(oppoIndex >= 0 && oppoIndex<nPoints);
			double xVal = (oppoPoint - xPoints[oppoIndex]).norm();
			double pVal = (pTexture[tll[s]] - pTexture[oppoIndex]).norm();
			if (xVal < miuX && pVal < niuP)
				lm.insert(tll[s]);
		}




		for (int s = 0; s < nPoints; s++)
		{
		/*	int tInx = -1;
			double xMin = -1;
#pragma omp parallel for
			for (int t = 0; t < inxlc; t++){
				disTmp[t] = (xPoints[s] - (ImNormalM2*xPoints[tlc[t]] + dnormalMd)).norm();				
			}
			for (int t = 0; t < inxlc; t++)
			{
				if (disTmp[t] < xMin || xMin < 0)
				{
					xMin = disTmp[t];
					tInx = t;
				}
			}*/
			resultSet.init(&retIndex[0], &outDistSqr[0]);
			Eigen::Vector3d oppoPoint = (ImNormalM2*xPoints[s] + dnormalMd);
			double queryPt[3] = { oppoPoint(0), oppoPoint(1), oppoPoint(2) };
			lcKDtree->findNeighbors(resultSet, &queryPt[0], nanoflann::SearchParams(10));
			int oppoIndex = tlc[retIndex[0]];

			double pVal = (pTexture[s] - pTexture[oppoIndex]).norm();
			double xVal = (oppoPoint - xPoints[oppoIndex]).norm();
			assert(oppoIndex >= 0&&oppoIndex<nPoints);
			if (xVal < miuX && pVal < niuP)
			{
				for (int i = 0; i < iLim; i++){
					pCandidates[i + iLim][s] = pCandidates[i][oppoIndex];
				}
			}
			if (int(s*100.0 / nPoints) > lastPro)
			{
				lastPro = int(s*100.0 / nPoints);
				printf("Final Pass %d %%\n", lastPro);
			}
		}
		for (std::set<int>::iterator it = lm.begin(); it != lm.end(); it++)
		{
			lc.insert(*it);
			int quan=ll.erase(*it);
			assert(1 == quan);
		}
	}
}
class MRFData{
public :
	TypeGeneral::REAL unaryEnergy[nPointsMax][nIteration];
	TypeGeneral::REAL edgeEnergy[nPointsMax][kapa][nIteration*nIteration];
};
MRFData *mData;
class pair_comparator : public stdext::hash_compare<std::pair<size_t, size_t> >{
	typedef std::pair<size_t, size_t> Key;
public:
	size_t operator()(const Key &key) const { return key.first*key.second; }
	bool operator()(const Key &first,
		const Key &second) const {
		return first.first != second.first ? first.first < second.first : first.second < second.second;
	}
};
std::hash_set<std::pair<int, int>, pair_comparator > relation;
int label[nPointsMax];

void outputResult();
void outputAttr();

void computeMRF(){
	MRFEnergy<TypeGeneral>* mrf;
	MRFEnergy<TypeGeneral>::NodeId* nodes;
	MRFEnergy<TypeGeneral>::Options options;
	TypeGeneral::REAL energy, lowerBound;

	const int nodeNum = nPoints; // number of nodes

	mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
	nodes = new MRFEnergy<TypeGeneral>::NodeId[nodeNum]; 
	mData=new MRFData();
	memset(mData->unaryEnergy, 0, sizeof(mData->unaryEnergy));
	memset(mData->edgeEnergy, 0, sizeof(mData->edgeEnergy));
	for (int i = 0; i < nPoints; i++)
		for (int j = 0; j < nIteration; j++)
		{
			if (-1 == pCandidates[j][i] )
				mData->unaryEnergy[i][j] = 10000;
			else
			{
				if (lv.count(i)>0 && 0 == j)
					mData->unaryEnergy[i][j] = zeta;
				else
					mData->unaryEnergy[i][j] = 1;
			}				
		}
	for (int i = 0; i < nPoints; i++)
		nodes[i] = mrf->AddNode(TypeGeneral::LocalSize(nIteration), TypeGeneral::NodeData(mData->unaryEnergy[i]));
	// construct energy
	const size_t numResults = kapa+1;
	for (int i = 0; i < nPoints; i++){
		double queryPt[3] = { xPointsCloud.pts[i].x, xPointsCloud.pts[i].y, xPointsCloud.pts[i].z };
		size_t retIndex[numResults];
		double outDistSqr[numResults];
		nanoflann::KNNResultSet<double> resultSet(numResults);
		resultSet.init(&retIndex[0], &outDistSqr[0]);
		pointKDtree->findNeighbors(resultSet, &queryPt[0], nanoflann::SearchParams(10));
		int passJS = 0;
		for (int no = 0;  no< numResults; no++){
			int j = retIndex[no];
			if (i == j || passJS>=kapa) continue;
			for (int ki = 0; ki < nIteration; ki++)
				for (int kj = 0; kj < nIteration; kj++){
					mData->edgeEnergy[i][passJS][ki*nIteration + kj] = 10000;
					if (pCandidates[ki][i] != -1 && pCandidates[kj][j] != -1){
						double tmp = (pTexture[pCandidates[ki][i]] - pTexture[pCandidates[kj][j]]).norm();
						tmp *= tmp;
						if (ki == kj)
							tmp *= beta;
						mData->edgeEnergy[i][passJS][ki*nIteration + kj] = tmp;

					}
				}
			assert(i != j);
			passJS++;
			mrf->AddEdge(nodes[i], nodes[j], TypeGeneral::EdgeData(TypeGeneral::GENERAL, mData->edgeEnergy[i][passJS]));
			relation.insert(std::make_pair(i, j));
			relation.insert(std::make_pair(j, i));
		}
	}
	options.m_iterMax = 30; // maximum number of iterations
	//options.m_printMinIter = 1;
	//options.m_printIter = 1;
	mrf->Minimize_TRW_S(options, lowerBound, energy);

	// read solution
	for (int i = 0; i < nPoints;i++)
		label[i] = mrf->GetSolution(nodes[i]);
	//outputResult();

	outputAttr();

	time_counter.update();
	printf("\nTime: %lf\n", time_counter.elapsed_time());
	printf("Releasing!");
	delete mData;
	delete nodes;
	delete mrf;
	printf("Finish!");	
}
void outputAttr(){
	for (int i = 0; i < nPoints; i++){
		int tmp = pCandidates[label[i]][i];
		if (-1 != tmp)
			pResult[i] = pTexture[tmp];
		else
			pResult[i] << -1, -1, -1;
	}
	FILE *f;
	fopen_s(&f, "result/attr_r.txt", "w");
	fprintf_s(f, "%d\n", nPoints);
	for (int i = 0; i < nPoints; i++)
	{
		for (int j = 0; j < nColorDim; j++)
			fprintf_s(f, "%lf ", pResult[i](j));
		fprintf_s(f, "\n");
	}
	fclose(f);
}

void outputResult(){
	for (int i = 0; i < nPoints; i++){
		int tmp = pCandidates[label[i]][i];
		if (-1 != tmp)
			pResult[i] = pTexture[tmp];
		else
			pResult[i] << -1, -1, -1;
	}
	export_pointcloud_ply("cube-result.ply", xPoints, nPoints, NULL, pResult);
}

void outputOrigin(){
	memset(label, 0, sizeof(label));
	for (int i = 0; i < nPoints; i++){
		int tmp = pCandidates[label[i]][i];
		if (-1 != tmp)
			pResult[i] = pTexture[tmp];
		else
			pResult[i] << 0, 0, 0;
	}
	export_pointcloud_ply("cube-origin.ply", xPoints, nPoints, NULL, pResult);
}
void freeResource(){
	delete pointKDtree;
	delete lcKDtree;
}

int main(){
	time_counter.update();

	readPly();
	//generateCube();
	//outputOrigin();

	generateKDTree();
	computeSymmetry();
	computeMRF();
	//outputResult();
	freeResource();
	system("pause");
	return 0;
}
void export_pointcloud_ply(const char *filename, Eigen::Vector3d *pos, int n,
	const Eigen::Vector3d *p_normal,
	const Eigen::Matrix<double, nColorDim, 1> *p_color)
{
	FILE *fp;

	fopen_s(&fp,filename, "wt");

	fprintf_s(fp, "ply\n");
	fprintf_s(fp, "format ascii 1.0\n");
	fprintf_s(fp, "comment (C) Hongzhi Wu, Sep 2013.\n");

	fprintf_s(fp, "element vertex %d\n", n);
	fprintf_s(fp, "property float x\n");
	fprintf_s(fp, "property float y\n");
	fprintf_s(fp, "property float z\n");
	if (p_normal)
	{
		fprintf_s(fp, "property float nx\n");
		fprintf_s(fp, "property float ny\n");
		fprintf_s(fp, "property float nz\n");
	}
	if (p_color)
	{
		fprintf_s(fp, "property uchar red\n");
		fprintf_s(fp, "property uchar green\n");
		fprintf_s(fp, "property uchar blue\n");
		fprintf_s(fp, "property uchar alpha\n");
	}

	fprintf_s(fp, "end_header\n");

	for (int i = 0; i < n; i++)
	{
		fprintf_s(fp, "%g %g %g ", pos[i](0), pos[i](1), pos[i](2));

		if (p_normal)
		{
			fprintf_s(fp, "%g %g %g ", (p_normal)[i](0), (p_normal)[i](1), (p_normal)[i](2));
		}

		if (p_color)
		{
			int r = std::max(std::min(int((p_color)[i](0) * 255), 255), 0),
				g = std::max(std::min(int((p_color)[i](1) * 255), 255), 0),
				b = std::max(std::min(int((p_color)[i](2) * 255), 255), 0);
			fprintf_s(fp, "%d %d %d %d ", r, g, b, 255);
		}

		fprintf_s(fp, "\n");
	}

	fclose(fp);
}

//-------set
//deep copy
//-------Eigen
// deep copy
// norm() 

//----------MAIN DEBUG

/*std::cout << relation.count(std::make_pair(0, 0)) << "\n";
std::cout << relation.count(std::make_pair(0, 39)) << "\n";
relation.insert(std::make_pair(0, 0));
//relation.insert(std::make_pair(1, 2));
//relation.insert(std::make_pair(600000, 1000000));
std::cout << relation.count(std::make_pair(0, 0)) << "\n";
std::cout << relation.count(std::make_pair(0, 39)) << "\n";
//std::cout << relation.count(std::make_pair(600000,1000000)) << "\n";
system("pause");*/
//std::cout << Eigen::Matrix3d::Identity() << "\n";
/*Eigen::Vector3d normalR(1, 2, 3);
Eigen::Vector3d normalR2;
normalR2 = normalR;
normalR(0) = 2;
std::cout << normalR2;*/
/*std::set<int>a, b;
a.insert(1);
a.insert(2);
a.insert(3);
a.insert(4);
b.insert(2);
b.insert(3);
b.insert(6);
b.insert(a.begin(), a.end());
for (std::set<int>::iterator it = b.begin(); it != b.end(); it++)
std::cout << *it << "\n";*/

//------KDTREE

/*const size_t numResults = kapa;
double queryPt[3] = { 1.0, 0.1, 0.1 };
size_t retIndex[numResults];
double outDistSqr[numResults];
nanoflann::KNNResultSet<double> resultSet(numResults);
resultSet.init(&retIndex[0], &outDistSqr[0]);
pointKDtree->findNeighbors(resultSet, &queryPt[0], nanoflann::SearchParams(10));
std::cout << "knnSearch(nn=" << numResults << "): \n";
for (int i = 0; i < numResults; i++)
{
std::cout << "ret_index=" << retIndex[i] << " out_dist_sqr=" << outDistSqr[i] << std::endl;
std::cout << xPointsCloud.pts[retIndex[i]].x << " " << xPointsCloud.pts[retIndex[i]].y << " " << xPointsCloud.pts[retIndex[i]].z << "\n";
}*/