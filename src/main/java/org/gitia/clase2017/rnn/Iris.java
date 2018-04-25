/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.gitia.clase2017.rnn;

import org.gitia.jdataanalysis.CSV;
import org.ejml.simple.SimpleMatrix;
import org.gitia.froog.Feedforward;
import org.gitia.froog.layer.Layer;
import org.gitia.froog.lossfunction.LossFunction;
import org.gitia.froog.statistics.Compite;
import org.gitia.froog.statistics.ConfusionMatrix;
import org.gitia.froog.trainingalgorithm.Backpropagation;
import org.gitia.froog.transferfunction.TransferFunction;

/**
 *
 * @author Matías Rodschild <mroodschild@gmail.com>
 */
public class Iris {
    public static void main(String[] args) {
        
        SimpleMatrix X = CSV.open("src/main/resources/iris/iris-in.csv").transpose();
        SimpleMatrix T = CSV.open("src/main/resources/iris/iris-out.csv").transpose();

        Feedforward net = new Feedforward();
        net.addLayer(new Layer(4, 10, TransferFunction.TANSIG));
        net.addLayer(new Layer(10, 3, TransferFunction.LOGSIG));

        Backpropagation bp = new Backpropagation();

        bp.setEpoch(1000);
        bp.setMomentum(0.9);
        bp.setLossFunction(LossFunction.MSE);

        bp.train(net, X, T);
        
        SimpleMatrix salidaNet = Compite.eval(net.output(X).transpose());

        System.out.println("\nMatriz de Confusion 1 - Datos de entrenamiento");
        ConfusionMatrix cmatrix = new ConfusionMatrix();
        cmatrix.eval(salidaNet, T.transpose());
        cmatrix.printStats();
    }
}
