/******************************************************************
 * Programa que genera una onda senoidal usando una red neuronal
 * MLP superficial
 * Autor:
 *      Jesús Alfonso López 
 *      jalopez@uao.edu.co
 ******************************************************************/

#include <math.h>

/******************************************************************
 * Definición estructura de la red
 ******************************************************************/

const int HiddenNodes = 10;
const int InputNodes = 4;
const int OutputNodes = 1;
const int DataVal = 50;
int Samples=0;

const float YMin = -1;
const float YMax = 1;

// Valores para normalizar la Entrada de la red  
const float X1Min = 1;
const float X1Max = 5;
const float X2Min = 1;
const float X2Max = 5;
const float X3Min = 3;
const float X3Max = 6;
const float X4Min = 1;
const float X4Max = 5;

// Valores para normalizar la salida de la red
const float DMin =0.13125;
const float DMax =7.10011;

// Los pesos de la red fueron obtenidos en TensorFlow y se copiaron a este programa
// Pesos capa oculta
const float HiddenWeights[HiddenNodes][InputNodes+1]= {
{1.3870015144348145, -1.5787280797958374, -0.6152315139770508, 0.10216180235147476, -4.051407814025879},
{1.727260708808899, -0.5600273013114929, 2.315542697906494, -0.6218500733375549, 0.8876901268959045},
{1.265161156654358, 2.5358669757843018,  0.3963158130645752,  3.2382802963256836,  2.6321394443511963},
{-0.6870208978652954, -0.34613874554634094,  3.409726142883301, 1.649113416671753, 3.1624398231506348},
{-1.9843926429748535, 0.3911074995994568,  -1.2769430875778198, 0.8291241526603699,  -1.3173857927322388},
{0.5212803483009338,  1.498855710029602, -1.1389256715774536, -1.497573971748352,  -1.5094305276870728},
{0.4571288228034973,  0.522105872631073, -2.3912289142608643, 0.7533050775527954,  -1.0125892162322998},
{3.4706342220306396,  -0.7483362555503845, -0.37019234895706177,  -1.201735496520996,  -3.1369030475616455},
{-5.425341606140137,  0.23286138474941254, -1.8912129402160645, 2.4139809608459473,  4.4206223487854},
{-0.5981503129005432, -2.276771068572998, -3.5300071239471436, -1.6594200134277344, -2.83518648147583}

    }; 

// Pesos capa de salida
const float OutputWeights[OutputNodes][HiddenNodes+1]  = {
{3.9525146484375, 2.2322425842285156,  -0.9860675930976868, -2.237312078475952,  2.2410144805908203,  -1.4054594039916992, 2.5653703212738037,  -2.977325439453125,  -1.3571876287460327, -0.9199329614639282, 0.8430728316307068

}
}; 

//Vector entrada A
const float A[1][DataVal]= {5, 5, 3, 2, 3, 3, 3, 3, 3, 3, 4, 1, 3, 5, 4, 5, 4, 1, 2, 1, 1, 2, 3, 5, 2, 1, 1, 1, 5, 2, 4, 1, 5, 2, 5, 4, 5, 1, 1, 5, 1, 4, 1, 4, 3, 2, 2, 3, 1, 1};
const float B[1][DataVal]= {4, 1, 4, 2, 1, 1, 5, 1, 1, 3, 1, 2, 3, 4, 3, 5, 2, 4, 2, 2, 3, 5, 2, 5, 5, 5, 2, 3, 3, 1, 4, 3, 1, 1, 5, 5, 2, 4, 4, 2, 2, 2, 1, 1, 5, 3, 5, 4, 2, 1};
const float C[1][DataVal]= {6, 4, 3, 4, 4, 5, 3, 3, 4, 3, 5, 5, 3, 5, 4, 6, 4, 4, 6, 4, 5, 4, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 4, 5, 3, 3, 3, 4, 3, 4, 3, 4, 5, 3, 6, 3, 3, 4, 3, 6};
const float D[1][DataVal]= {5, 2, 1, 1, 2, 1, 1, 1, 3, 1, 4, 4, 2, 4, 3, 5, 2, 2, 5, 1, 4, 1, 1, 2, 3, 1, 2, 1, 2, 1, 2, 2, 1, 4, 1, 2, 1, 1, 1, 3, 1, 3, 4, 2, 5, 2, 1, 3, 2, 5};


int i, j, p, q, r;
float Accum;
float Hidden[HiddenNodes];
float Output[OutputNodes];
float Input[InputNodes];
 

void setup(){
  //start serial connection
  Serial.begin(9600);
}

void loop(){
  
  float Entrada;
  float Salida;
  float Tiempo;


if (Samples<50)
{
  Input[0]=A[0][Samples];
  Input[1]=B[0][Samples];
  Input[2]=C[0][Samples];
  Input[3]=D[0][Samples];
  Samples=Samples+1;
}

if (Samples==50)
{
  Samples=0;
}

// Normalización de las entradas que se usará en la red neuronal
Input[0]=YMin+ (2*((Input[0]-X1Min)/(X1Max-X1Min)));
Input[1]=YMin+ (2*((Input[1]-X2Min)/(X2Max-X2Min)));
Input[2]=YMin+ (2*((Input[2]-X3Min)/(X3Max-X3Min)));
Input[3]=YMin+ (2*((Input[3]-X4Min)/(X4Max-X4Min)));

/******************************************************************
* Cálculo de la salida de capa oculta
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[i][InputNodes] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += HiddenWeights[i][j]*Input[j];
      }
      // función de activación sigmoidal 
        Hidden[i] = (1)/(1+exp(-Accum));
    }

/******************************************************************
* Cálculo de la salida de capa de salida
******************************************************************/
    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[i][HiddenNodes] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum +=  OutputWeights[i][j]*Hidden[j];
      }
      // función de activación lineal
        Output[i] = Accum; 

    }
Salida=Output[0];
// Desnormalización de la salida de la red neuronal
Salida=((DMax-DMin)*((Salida+1)/2))+DMin;


Serial.print("Salida Des: "); 
Serial.println( Salida);      

delay(50);  

}



