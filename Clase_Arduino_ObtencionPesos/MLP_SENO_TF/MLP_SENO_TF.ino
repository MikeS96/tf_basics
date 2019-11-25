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
const int InputNodes = 1;
const int OutputNodes = 1;

// Los pesos de la red fueron obtenidos en TensorFlow y se copiaron a este programa
// Pesos capa oculta
const float HiddenWeights[HiddenNodes][InputNodes+1]= {
{-0.363107323647,  -0.50464951992},
{-0.598850727081, 0.832224607468},
{0.40849390626, 0.0688511133194},
{0.414619863033,  -2.53732132912},
{-0.607468903065, 2.36923527718},
{-0.420356690884, -0.937813580036},
{1.00892925262, -0.16453525424},
{-0.337994396687, 2.01401972771},
{0.722436368465,  -0.833146035671},
{0.614956378937,  -1.82312440872}


    }; 

// Pesos capa de salida
const float OutputWeights[OutputNodes][HiddenNodes+1]  = {
{0.364980012178,0.780598759651,-0.264719724655,1.85506165028,1.33418726921,-0.214293897152,0.9527541399,-1.42306041718,1.14332330227,-1.20784533024,1.18532776833

}
    }; 

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


Tiempo=millis();
//Artificio para generar una entrada entre 0 y 2*pi
Entrada=(fmod(Tiempo,6283))/1000; 


Input[0]=Entrada;
/******************************************************************
* Cálculo de la salida de capa oculta
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[i][InputNodes] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += HiddenWeights[i][j]*Input[j];
      }
      // función de activación tangente hiperbólica
        Hidden[i] = (exp(Accum)-exp(-Accum))/(exp(Accum)+exp(-Accum));
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


Serial.print("Salida Des: "); 
Serial.println( Salida);      

delay(50);  

}



