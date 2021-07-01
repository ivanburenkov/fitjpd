#include "stdio.h"
#include "stdlib.h"
#include <math.h>
#include </home/ivan/js/gsl/gsl-js-master/gsl/gsl_rng.h>
#include </home/ivan/js/gsl/gsl-js-master/gsl/gsl_vector.h>
#include </home/ivan/js/gsl/gsl-js-master/gsl/gsl_randist.h>
#include </home/ivan/js/gsl/gsl-js-master/gsl/gsl_vector.h>
#include </home/ivan/js/gsl/gsl-js-master/gsl/gsl_blas.h>
#include </home/ivan/js/gsl/gsl-js-master/gsl/gsl_multifit_nlin.h>
#include "emscripten.h"
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Thermal mode distribution
//mu - average phtotn number [0.0 ... MAX_DOUBLE]
//*P - distrbution array
//nmax - size of disstribution array (max phtotn number-1)
EMSCRIPTEN_KEEPALIVE
int Pt(double mu, double *P, int nmax){
	int k;
	for(k=0;k<nmax;k++){
	P[k]=1/(1 + mu)*pow((mu/(1 + mu)),k);
	}
	return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Poisson mode distribution
//mu - average phtotn number [0.0 ... MAX_DOUBLE]
//*P - distrbution array
//nmax - size of disstribution array (max phtotn number-1)
EMSCRIPTEN_KEEPALIVE
int Pp(double mu, double *P, int nmax){
	int k;
  P[0]=exp(-mu);
	for(k=1;k<nmax;k++){
	P[k]=P[k-1]*mu/k;
	}
	return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Single photon mode distribution
//mu - average phtotn number [0.0 ... 1.0]
//*P - distrbution array
//nmax - size of disstribution array (max phtotn number-1)
EMSCRIPTEN_KEEPALIVE
int Ps(double mu, double *P, int nmax){
	int k;
  P[0]=1.0-mu;
  P[1]=mu;	
  for(k=2;k<nmax;k++){
	P[k]=0.0;
	}
	return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Vacuum mode distribution
//*P - distrbution array
//nmax - size of disstribution array (max phtotn number-1)
EMSCRIPTEN_KEEPALIVE
int P0(double *P, int nmax){
	int k;
  P[0]=1.0;
  for(k=1;k<nmax;k++){
	P[k]=0.0;
	}
	return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
EMSCRIPTEN_KEEPALIVE
int Max(int a, int b){
	if(a>=b){
		return a;
	} else {
		return b;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
EMSCRIPTEN_KEEPALIVE
int Min(int a, int b){
	if(a<=b){
		return a;
	} else {
		return b;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Add another mode distribution PM to existiong mode distribution P
//*P - distrbution array
//*PM - distrbution array
//nmax - size of disstribution array (max phtotn number-1)
EMSCRIPTEN_KEEPALIVE
int PAdd(double *P, double *PM, int nmax){
  int nn,i;
  int nm=nmax;
  double * curP=(double *)malloc(nm*sizeof(double)); 
	for (nn=0;nn<nm;nn++){
    curP[nn]=0.0;
    for(i=0;i<=nn;i++){
      curP[nn]+=P[i]*PM[nn-i];
    }
  }    
  for (nn=0;nn<nm;nn++){
    P[nn]=curP[nn];
  }
  free(curP);
	return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Loss factors
//eta - efficiency [0.0 ... 1.0] (1-Loss)
//*Loss - loss factors array
//nm - size of disstribution array (max phtotn number-1)
//nmax - (max phtotn number-1) state to contribute in distribution due to losses 
EMSCRIPTEN_KEEPALIVE
int Losses(double eta, double *Loss, int nm, int nmax){
	int n,ipk,q,p;
#ifdef _OPENMP
#pragma omp parallel for private(n,ipk,q,p)
#endif
	for(n=0;n<nm;n++){
		for(ipk=n;ipk<2*nmax;ipk++){
			Loss[n*2*nmax+ipk]=1.0;
			for(q=0;q<ipk-n;q++){
				Loss[n*2*nmax+ipk] *= (1.0-eta);
			}
			for(p=0;p<n;p++){
				Loss[n*2*nmax+ipk] *= eta*(ipk-n+p+1.0)/(p+1.0);
			}
			//printf("%d\t%d\t%.20g\n",n,ipk,LossL[n*2*nmax+ipk]);
		}
	}
	return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Generate JPD
//*x vector of parameters of lenght (\Sum pp[i]) + 2 [Number of modes + 2 efficiency losses]  
//pp[9] matrix of mode numbers vs arms (Conjugated,Signal,Idler) vs type (Thermal,Single Photon,Poisson) 
//nn - size of JPD matrix nn*nn
//*z - JPD array of length nn*nn
EMSCRIPTEN_KEEPALIVE
int gen_jpd_z (double * x_init, int * pp, int nn, double * z){

//Total number of fitting (generating) parameters
  int p = pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5]+pp[6]+pp[7]+pp[8]+2;
//Number of states to contubute in JPD due to the losses
	int nmax=3*nn;
//Counters
  int i,ii,n,m,j,k,s,q,r,ipk;
//Array of parameters
  double *mu=(double *)malloc((p-2)*sizeof(double));
  for(j=0;j<p-2;j++){
    mu[j]=x_init[j];
  }
  double etaL=x_init[p-2];
  double etaR=x_init[p-1];
//Raw probability distributions for All represented modes
  double *PM = (double *)malloc(((pp[0]+pp[1]+pp[2])*nmax+(p-2-pp[0]-pp[1]-pp[2])*nn)*sizeof(double));
//Correlated and Signal/Idler PDs
  double *P_c = (double *)malloc(nmax*sizeof(double));
  double *P_s = (double *)malloc(nn*sizeof(double));
  double *P_i = (double *)malloc(nn*sizeof(double));
  double *LossL = (double *)malloc(2*nmax*nn*sizeof(double));
	double *LossR = (double *)malloc(2*nmax*nn*sizeof(double));
//Initialize PDs
  P0(P_c,nmax);
  P0(P_s,nn);
  P0(P_i,nn);

  int pc=0;
//Add correlated modes
  for(j=0;j<pp[0];j++){
    //Thermal
    Pt(mu[j+pc],PM+(j+pc)*nmax,nmax);
    PAdd(P_c,PM+(j+pc)*nmax,nmax);
  }
  pc+=pp[0];
  for(j=0;j<pp[1];j++){
    //Single photon
    Ps(mu[pc+j],PM+(pc+j)*nmax,nmax);
    PAdd(P_c,PM+(j+pc)*nmax,nmax);
  }
  pc+=pp[1];
  for(j=0;j<pp[2];j++){
    //Poissonian
    Pp(mu[pc+j],PM+(pc+j)*nmax,nmax);
    PAdd(P_c,PM+(j+pc)*nmax,nmax);
  }
  pc+=pp[2];
//Add uncorrelated signal modes
  for(j=0;j<pp[3];j++){
    //Thermal
    Pt(mu[j+pc],PM+(j+pc)*nn,nn);
    PAdd(P_s,PM+(j+pc)*nn,nn);
  }
  pc+=pp[3];
  for(j=0;j<pp[4];j++){
    //Single photon
    Ps(mu[pc+j],PM+(pc+j)*nn,nn);
    PAdd(P_s,PM+(j+pc)*nn,nn);
  }
  pc+=pp[4];
  for(j=0;j<pp[5];j++){
    //Poissonian
    Pp(mu[pc+j],PM+(pc+j)*nn,nn);
    PAdd(P_s,PM+(j+pc)*nn,nn);
  }
  pc+=pp[5];
//Add uncorrelated idler modes
  for(j=0;j<pp[6];j++){
    //Thermal
    Pt(mu[j+pc],PM+(j+pc)*nn,nn);
    PAdd(P_i,PM+(j+pc)*nn,nn);
  }
  pc+=pp[6];
  for(j=0;j<pp[7];j++){
    //Single photon
    Ps(mu[pc+j],PM+(pc+j)*nn,nn);
    PAdd(P_i,PM+(j+pc)*nn,nn);
  }
  pc+=pp[7];
  for(j=0;j<pp[8];j++){
    //Poissonian
    Pp(mu[pc+j],PM+(pc+j)*nn,nn);
    PAdd(P_i,PM+(j+pc)*nn,nn);
  }
  
//Calculate Loss Factors
  Losses(etaL,LossL,nn,nmax);
	Losses(etaR,LossR,nn,nmax);

  double sum;
//Conjugated part of JPD
  double *P_cl = (double *)malloc(nn*nn*sizeof(double));

//JPD for conjugated part, including losses
#ifdef _OPENMP
#pragma omp parallel for private(n,m,k)
#endif
	for(n=0;n<nn;n++){
    for(m=0;m<nn;m++){
      P_cl[n*nn+m]=0.0;
      for(k=Max(m,n);k<nmax;k++){
        P_cl[n*nn+m]+=P_c[k]*LossL[n*2*nmax+k]*LossR[m*2*nmax+k];
			}
    }
  }
//Total JPD, losses are not considered for the background
#ifdef _OPENMP
#pragma omp parallel for private(n,m,k,j,i,ii,sum)
#endif
	for(n=0;n<nn;n++){
    for(m=0;m<nn;m++){
      sum=0.0;
      for(j=0;j<=m;j++){
			  for(i=0;i<=n;i++){
          sum+=P_s[i]*P_i[j]*P_cl[(n-i)*nn+m-j];
				}
			}
			ii=n*nn+m;
 			z[ii]=sum;
    }
	}

  free(PM);
  free(P_c);
  free(P_cl);
  free(P_i);
  free(P_s);
  free(mu);
  free(LossL);
  free(LossR);

	return GSL_SUCCESS;
}
///////
EMSCRIPTEN_KEEPALIVE
//double* genJPD(int *pp, double *x_init, double *doubleVector, int len) {
void genJPD(int *pp, double *x_init, double *doubleVector, int len) {

   gen_jpd_z(x_init, pp, len, doubleVector);
   //return doubleVector;
}
///////
EMSCRIPTEN_KEEPALIVE
double* jpd2rpd(double *y, double *rpd, int len){
  double sum;
  int i,j;
  int n=len;
  //Calculate and write in file Signal RPD
  for(i=0;i<n;i++){
    sum=0.0;
  	for(j=0;j<n;j++){
      sum+=y[n*i+j];
    }
		rpd[i]=sum;
	}
	
  //Calculate and write in file Idler RPD
  for(i=0;i<n;i++){
    sum=0.0;
  	for(j=0;j<n;j++){
      sum+=y[n*j+i];
    }
		rpd[i+n]=sum;
	}
	
  return rpd;
}
EMSCRIPTEN_KEEPALIVE
double* normdata(double *z, double *zn, int len){
  double min=1.0;
  double max=0.0;
  int i;
  for(i=0;i<len*len;i++){
    if(min>z[i])min=z[i];
    if(max<z[i])max=z[i];
  }
  for(i=0;i<len*len;i++){
    zn[i]=(z[i]-min)/(max-min);
  }
  return zn;
}
EMSCRIPTEN_KEEPALIVE
int SC(int i, int rgb){
double sc[256][3]={{0., 0., 0.}, {0.0087716, 0.00319529, 0.0119177}, {0.0175432, 0.00639059, 0.0238354}, {0.0263148, 0.00958588, 0.0357532}, {0.0350864, 0.0127812, 0.0476709}, {0.043858, 0.0159765,0.0595886}, {0.0526296, 0.0191718, 0.0715063}, {0.0614012, 0.0223671, 0.083424}, {0.0701728, 0.0255624, 0.0953417}, {0.0789444,0.0287576, 0.107259}, {0.087716, 0.0319529, 0.119177}, {0.0964876, 0.0351482, 0.131095}, {0.105259, 0.0383435, 0.143013}, {0.114031, 0.0415388, 0.15493}, {0.122802, 0.0447341, 0.166848}, {0.131574, 0.0479294, 0.178766}, {0.140346, 0.0511247, 0.190683}, {0.149117, 0.05432, 0.202601}, {0.157889, 0.0575153, 0.214519}, {0.16666, 0.0607106, 0.226437}, {0.175432, 0.0639059, 0.238354}, {0.184204, 0.0671012, 0.250272}, {0.192975, 0.0702965, 0.26219}, {0.201747, 0.0734918, 0.274108}, {0.210518, 0.0766871, 0.286025}, {0.21929, 0.0798824, 0.297943}, {0.228062, 0.0830776, 0.309861}, {0.236833, 0.0862729, 0.321778}, {0.245605, 0.0894682, 0.333696}, {0.254376, 0.0926635, 0.345614}, {0.263148, 0.0958588, 0.357532}, {0.27192, 0.0990541, 0.369449}, {0.280691, 0.102249, 0.381367}, {0.289463, 0.105445, 0.393285}, {0.298234, 0.10864, 0.405202}, {0.307006, 0.111835, 0.41712}, {0.315778, 0.115031, 0.429038}, {0.324549, 0.118226, 0.440956}, {0.333321, 0.121421, 0.452873}, {0.342092, 0.124616, 0.464791}, {0.350864, 0.127812, 0.476709}, {0.359636, 0.131007, 0.488626}, {0.368407, 0.134202, 0.500544}, {0.377681, 0.137259, 0.50373}, {0.387457, 0.140177, 0.498183}, {0.397234, 0.143095, 0.492637}, {0.40701, 0.146013, 0.48709}, {0.416786, 0.148931, 0.481544}, {0.426563, 0.151849, 0.475997}, {0.436339, 0.154767, 0.470451}, {0.446115, 0.157685, 0.464904}, {0.455892, 0.160603, 0.459358}, {0.465668, 0.163521, 0.453812}, {0.475444, 0.166439, 0.448265}, {0.485221, 0.169357, 0.442719}, {0.494997, 0.172275, 0.437172}, {0.504773, 0.175193, 0.431626}, {0.51455, 0.178111, 0.426079}, {0.524326, 0.181029, 0.420533}, {0.534102, 0.183947, 0.414986}, {0.543879, 0.186865, 0.40944}, {0.553655, 0.189783, 0.403893}, {0.563431, 0.192701, 0.398347}, {0.573208, 0.195619, 0.3928}, {0.582984, 0.198538, 0.387254}, {0.59276, 0.201456, 0.381707}, {0.602537, 0.204374, 0.376161}, {0.612313, 0.207292, 0.370614}, {0.622089, 0.21021, 0.365068}, {0.631866, 0.213128, 0.359522}, {0.641642, 0.216046, 0.353975}, {0.651418, 0.218964, 0.348429}, {0.661195, 0.221882, 0.342882}, {0.670971, 0.2248, 0.337336}, {0.680747, 0.227718, 0.331789}, {0.690524, 0.230636, 0.326243}, {0.7003, 0.233554, 0.320696}, {0.710076, 0.236472, 0.31515}, {0.719853, 0.23939, 0.309603}, {0.729629, 0.242308, 0.304057}, {0.739405, 0.245226, 0.29851}, {0.749182, 0.248144, 0.292964}, {0.758958, 0.251062, 0.287417}, {0.768734, 0.25398, 0.281871}, {0.778511, 0.256898, 0.276324}, {0.788287, 0.259816, 0.270778}, {0.792783, 0.264325, 0.26561}, {0.797279, 0.268835, 0.260442}, {0.801776, 0.273344, 0.255274}, {0.806272, 0.277854, 0.250106}, {0.810768, 0.282363, 0.244937}, {0.815264, 0.286873, 0.239769}, {0.819761, 0.291382, 0.234601}, {0.824257, 0.295891, 0.229433}, {0.828753, 0.300401, 0.224265}, {0.833249, 0.30491, 0.219097}, {0.837746, 0.30942, 0.213929}, {0.842242, 0.313929, 0.208761}, {0.846738, 0.318439, 0.203592}, {0.851234, 0.322948, 0.198424}, {0.855731, 0.327458, 0.193256}, {0.860227, 0.331967, 0.188088}, {0.864723, 0.336476, 0.18292}, {0.869219, 0.340986, 0.177752}, {0.873715, 0.345495, 0.172584}, {0.878212, 0.350005, 0.167416}, {0.882708, 0.354514, 0.162247}, {0.887204, 0.359024, 0.157079}, {0.8917, 0.363533, 0.151911}, {0.896197, 0.368042, 0.146743}, {0.900693, 0.372552, 0.141575}, {0.905189, 0.377061, 0.136407}, {0.909685, 0.381571, 0.131239}, {0.914182, 0.38608, 0.126071}, {0.918678, 0.39059, 0.120903}, {0.923174, 0.395099, 0.115734}, {0.92767, 0.399608, 0.110566}, {0.932167, 0.404118, 0.105398}, {0.936663, 0.408627, 0.10023}, {0.941159, 0.413137, 0.0950619}, {0.945655, 0.417646, 0.0898938}, {0.950151, 0.422156, 0.0847257}, {0.954648, 0.426665, 0.0795576}, {0.959144, 0.431175, 0.0743894}, {0.96364, 0.435684, 0.0692213}, {0.968136, 0.440193, 0.0640532}, {0.972633, 0.444703, 0.0588851}, {0.977129, 0.449212, 0.053717}, {0.97962, 0.454187, 0.0520581}, {0.980105, 0.459628, 0.0539084}, {0.98059, 0.465068, 0.0557587}, {0.981075, 0.470509, 0.057609}, {0.981561, 0.475949, 0.0594593}, {0.982046, 0.48139, 0.0613096}, {0.982531, 0.48683, 0.0631599}, {0.983016, 0.492271, 0.0650102}, {0.983502, 0.497711, 0.0668605}, {0.983987, 0.503152, 0.0687108}, {0.984472, 0.508592, 0.0705611}, {0.984957, 0.514033, 0.0724114}, {0.985443, 0.519473, 0.0742618}, {0.985928, 0.524914, 0.0761121}, {0.986413, 0.530354, 0.0779624}, {0.986898, 0.535795, 0.0798127}, {0.987384, 0.541235, 0.081663}, {0.987869, 0.546676, 0.0835133}, {0.988354, 0.552116, 0.0853636}, {0.988839, 0.557557, 0.0872139}, {0.989325, 0.562997, 0.0890642}, {0.98981, 0.568438, 0.0909145}, {0.990295, 0.573878, 0.0927648}, {0.99078, 0.579319, 0.0946151}, {0.991266, 0.584759, 0.0964655}, {0.991751, 0.5902, 0.0983158}, {0.992236, 0.59564, 0.100166}, {0.992721, 0.601081, 0.102016}, {0.993207, 0.606521, 0.103867}, {0.993692, 0.611962, 0.105717}, {0.994177, 0.617402, 0.107567}, {0.994662, 0.622843, 0.109418}, {0.995148, 0.628283, 0.111268}, {0.995633, 0.633724, 0.113118}, {0.996118, 0.639164, 0.114969}, {0.996603, 0.644605, 0.116819}, {0.997089, 0.650045, 0.118669}, {0.997574, 0.655486, 0.120519}, {0.998059, 0.660926, 0.12237}, {0.998544, 0.666367, 0.12422}, {0.99903, 0.671807, 0.12607}, {0.999515, 0.677248, 0.127921}, {1., 0.682688, 0.129771}, {1., 0.687383, 0.138273}, {1., 0.692078, 0.146774}, {1., 0.696774, 0.155276}, {1., 0.701469, 0.163778}, {1., 0.706164, 0.17228}, {1., 0.710859, 0.180781}, {1., 0.715555, 0.189283}, {1., 0.72025, 0.197785}, {1., 0.724945, 0.206286}, {1., 0.72964, 0.214788}, {1., 0.734336, 0.22329}, {1., 0.739031, 0.231792}, {1., 0.743726, 0.240293}, {1., 0.748421, 0.248795}, {1., 0.753117, 0.257297}, {1., 0.757812, 0.265798}, {1., 0.762507, 0.2743}, {1., 0.767202, 0.282802}, {1., 0.771898, 0.291304}, {1., 0.776593, 0.299805}, {1., 0.781288, 0.308307}, {1., 0.785983, 0.316809}, {1., 0.790679, 0.325311}, {1., 0.795374, 0.333812}, {1., 0.800069, 0.342314}, {1., 0.804764, 0.350816}, {1., 0.80946, 0.359317}, {1., 0.814155, 0.367819}, {1., 0.81885, 0.376321}, {1., 0.823545, 0.384823}, {1., 0.828241, 0.393324}, {1., 0.832936, 0.401826}, {1., 0.837631, 0.410328}, {1., 0.842326, 0.418829}, {1., 0.847022, 0.427331}, {1., 0.851717, 0.435833}, {1., 0.856412, 0.444335}, {1., 0.861107, 0.452836}, {1., 0.865803, 0.461338}, {1., 0.870498, 0.46984}, {1., 0.875193, 0.478341}, {1., 0.879888, 0.486843}, {1., 0.883621, 0.497081}, {1., 0.886392, 0.509055}, {1., 0.889163, 0.52103}, {1., 0.891934, 0.533004}, {1., 0.894705, 0.544978}, {1., 0.897476, 0.556952}, {1., 0.900247, 0.568927}, {1., 0.903018, 0.580901}, {1., 0.905789, 0.592875}, {1., 0.90856, 0.604849}, {1., 0.911331, 0.616824}, {1., 0.914102, 0.628798}, {1., 0.916872, 0.640772}, {1., 0.919643, 0.652746}, {1., 0.922414, 0.664721}, {1., 0.925185, 0.676695}, {1., 0.927956, 0.688669}, {1., 0.930727, 0.700644}, {1., 0.933498, 0.712618}, {1., 0.936269, 0.724592}, {1., 0.93904, 0.736566}, {1., 0.941811, 0.748541}, {1., 0.944582, 0.760515}, {1., 0.947353, 0.772489}, {1., 0.950123, 0.784463}, {1., 0.952894, 0.796438}, {1., 0.955665, 0.808412}, {1., 0.958436, 0.820386}, {1., 0.961207, 0.83236}, {1., 0.963978, 0.844335}, {1., 0.966749, 0.856309}, {1., 0.96952, 0.868283}, {1., 0.972291, 0.880257}, {1., 0.975062, 0.892232}, {1., 0.977833, 0.904206}, {1., 0.980604, 0.91618}, {1., 0.983374, 0.928154}, {1., 0.986145, 0.940129}, {1., 0.988916, 0.952103}, {1., 0.991687, 0.964077}, {1., 0.994458, 0.976051}, {1., 0.997229, 0.988026}, {1., 1., 1.}};
return (int)(sc[i][rgb]*255);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//EMSCRIPTEN_KEEPALIVE
struct data {
	size_t n;
	double * y;
	double * sigma;
  int *pp;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Calculate JPD for parameters *x and returns fit errors *f for *data for GSL function gsl_multifit_fdfsolver
EMSCRIPTEN_KEEPALIVE
int mode_reconstruction_f (const gsl_vector * x, void *data, gsl_vector * f){
  size_t nnsq = ((struct data *)data)->n;
  int nn = (int)sqrt(nnsq);
	double *y = ((struct data *)data)->y;
	double *sigma = ((struct data *) data)->sigma;
  int *pp = ((struct data *)data)->pp;
//Total number of fitting (generating) parameters
  int p = pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5]+pp[6]+pp[7]+pp[8]+2;
//Array of parameters
  double *x_init=(double *)malloc(p*sizeof(double));
  int n,m,i,j;
  for(j=0;j<p;j++){
    x_init[j]=gsl_vector_get (x, j);
  }
//JPD array
  double *z = (double *)malloc(nn*nn*sizeof(double));
//Generage JPD
  gen_jpd_z(x_init, pp, nn, z);
//Calculated fit errors
	for(n=0;n<nn;n++){
    for(m=0;m<nn;m++){
      i=n*nn+m;
      //gsl_vector_set (f, i, (z[i] - y[i])/sigma[i]); //Linear scoring function
			gsl_vector_set (f, i, (pow(z[i],0.5) - pow(y[i],0.5))/sigma[i]);
    }
	}
  free(z);
 	return GSL_SUCCESS;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//double* genJPD(int *pp, double *x_init, double *doubleVector, int len)
EMSCRIPTEN_KEEPALIVE
void print_state (size_t iter, gsl_multifit_fdfsolver * s, int p){
	int i;
  printf ("iter: %3zu x =",iter);
  for(i=0;i<p;i++){
    printf("% 15.10f ",	gsl_vector_get (s->x, i));
  }
  printf("|f(x)| = %g\n",gsl_blas_dnrm2 (s->f));
}

EMSCRIPTEN_KEEPALIVE
double* fitJPD(int *pp, double *x_init, double *doubleVector, int len) {

  printf("len=%d\n",len);
  //total number of fit parameters (number of modes +2 efficiencies (1-loss))
  const size_t p = (size_t)(pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5]+pp[6]+pp[7]+pp[8]+2);
  
  
  double *jpd=doubleVector;
  printf("Total number of modes =%zu\n",p);
  
  double data;
	int ndata=len*len;
  double  mindata=1.0;
  double norm=0.0;
  
  int j,k;
  double * y = (double *)malloc(ndata*sizeof(double));
  double * sigma = (double *)malloc(ndata*sizeof(double));
  for(j=0;j<p;j++){
    x_init[j]=x_init[j]*0.5;
    printf("xinit[%d]=%lf\n",j,x_init[j]);
  }
  //x_init[0]=100.0;
  //Find minimum non-zero value to define accuracy
  //Number of trials calculated from minimum non-zero value, assuming it corresponds to single measurement event
  printf("ndata=%d\n",ndata);
  for(j=0;j<ndata;j++) {
    data=jpd[j];
    y[j] = data;
    //printf("data[%d]=%lf\n",j,data);
		//sigma[j] = (data>0.0)? sqrt(mindata/data):1.0;
    if((data<mindata)&&(data!=0.0)){
      mindata=data;
    }
		//ndata++;
    norm+=data;
	}
  //printf("ndata=%d\n",ndata);
  //printf("mindata = %0.15g\n Total numeber of trials = %g\n Norm = %g\n",mindata,1.0/mindata/norm,norm);
  for(j=0;j<ndata;j++) {
    sigma[j] = (y[j]>0)? sqrt(mindata/y[j]):100.0;
	}
  
  //Additional gsl parameters
  /*const gsl_rng_type * type;
  gsl_rng * r; 
  gsl_rng_env_setup();
	type = gsl_rng_default;
  r = gsl_rng_alloc (type);
  */

  const gsl_multifit_fdfsolver_type *T;
	gsl_multifit_fdfsolver *s;
	int status;
	unsigned int i, iter = 0;
	const size_t n = ndata; //total number of JPD values
  gsl_matrix *covar = gsl_matrix_alloc (p, p);
	
	struct data d = { n, y, sigma, pp};
  
	gsl_multifit_function_fdf f;
  	gsl_vector_view x = gsl_vector_view_array (x_init, p);
//Set fitting algorithm
	f.f = &mode_reconstruction_f; //Reference to the function used to calculate fit errors for the model
	f.df = 0; //&mode_reconstruction_df; -|- model Jacobian (NOT USED, Jacobian is internally computed using finite difference approximations of the function f, see GSL reference chapter 38.3)
	f.fdf = 0; //&mode_reconstruction_fdf; -|-
	f.n = n;
	f.p = p;
	f.params = &d;

	T = gsl_multifit_fdfsolver_lmsder;
	s = gsl_multifit_fdfsolver_alloc (T, n, p);
	gsl_multifit_fdfsolver_set (s, &f, &x.vector);
  //gsl_vector * g = gsl_vector_alloc( p );
  do{
		iter++;
		status = gsl_multifit_fdfsolver_iterate (s);
		printf ("status = %s\n", gsl_strerror (status));
  	print_state (iter, s,p);
		if (status)
		break;
    
    status = gsl_multifit_test_delta (s->dx, s->x, 1e-6, 1e-6);
    //gsl_multifit_gradient(s->J,s->f,g);
    //status = gsl_multifit_test_gradient (g, 1e-19);
    
	}
	while (status == GSL_CONTINUE && iter < 1000);

//Results output	
	gsl_multifit_covar (s->J, mindata, covar);
	#define FIT(i) gsl_vector_get(s->x, i)
	#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))
	{
		double chi = gsl_blas_dnrm2(s->f);
		double dof = n - p;
		double c = GSL_MAX_DBL(1, chi / sqrt(dof));
		printf("chisq/dof = %g\n", pow(chi, 2.0) / dof);
    for(i=0;i<p-2;i++){
      if(i<pp[0]){
        printf("mu%d (Thermal Conjugated) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      }
      if((pp[0]<=i)&&(i<pp[0]+pp[1])){
        printf("mu%d (Single Conjugated) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      } 
      if((pp[0]+pp[1]<=i)&&(i<pp[0]+pp[1]+pp[2])){
        printf("mu%d (Poisson Conjugated) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      }
      if((i>=pp[0]+pp[1]+pp[2])&&(i<pp[0]+pp[1]+pp[2]+pp[3])){
        printf("mu%d (Thermal Signal) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      }
      if((i>=pp[0]+pp[1]+pp[2]+pp[3])&&(i<pp[0]+pp[1]+pp[2]+pp[3]+pp[4])){
        printf("mu%d (Single Signal) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      } 
      if((i>=pp[0]+pp[1]+pp[2]+pp[3]+pp[4])&&(i<pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5])){
        printf("mu%d (Poisson Signal) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      }
      if((i>=pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5])&&(i<pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5]+pp[6])){
        printf("mu%d (Thermal Idler) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      }
      if((i>=pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5]+pp[6])&&(i<pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5]+pp[6]+pp[7])){
        printf("mu%d (Single Idler) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      } 
      if(pp[0]+pp[1]+pp[2]+pp[3]+pp[4]+pp[5]+pp[6]+pp[7]<=i){
        printf("mu%d (Poisson Idler) = % 15.10f +/-% 15.10f\n",i,FIT(i),c*ERR(i));
      }
    }
      printf("eta_Signal = % 15.10f +/-% 15.10f\n",FIT(p-2),c*ERR(p-2));
      printf("eta_Idler = % 15.10f +/-% 15.10f\n",FIT(p-1),c*ERR(p-1));
	}
  
  for(i=0;i<ndata;i++){
			jpd[i]=pow(gsl_vector_get(s->f, i)*sigma[i]+pow(y[i],0.5),2);
	}

  gsl_multifit_fdfsolver_free (s);
	gsl_matrix_free (covar);
	//gsl_rng_free (r);
  free(y);
  free(sigma);
  //free(x_init);
  
  return doubleVector;
}