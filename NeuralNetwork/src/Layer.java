import java.util.ArrayList;

public class Layer {

	private boolean DEBUG = true;
	
	private final int numNeurons;
	private final int numOfNNOutputs;
	private final double minGroundTruth;
	private final double maxGroundTruth;
	private final double midGroundTruth;
	
	private final double screenDistance = Math.sqrt(600*600 + 600*600);
	private double prevDistance = 0.0;
	
	private ArrayList<Neuron> neurons = new ArrayList<>();
//	private ArrayList<Double> inputs = new ArrayList<>();
	
	Layer(){
		numNeurons = 0;
		numOfNNOutputs = 0;
		minGroundTruth = 0.0;
		maxGroundTruth = 0.0;
		midGroundTruth = 0.0;
	}
	
	Layer(int numNeurons, int numInputs, double learningRate, int numOfNNOutputs, double minGroundTruth, double maxGroundTruth){
		
		this.numOfNNOutputs = numOfNNOutputs;
		this.minGroundTruth = minGroundTruth;
		this.maxGroundTruth = maxGroundTruth;
		this.midGroundTruth = maxGroundTruth/2;
		
		this.numNeurons = numNeurons;
				
		for(int cnt = 0; cnt < numNeurons; ++cnt){
			Neuron newNeuron = new Neuron(numInputs, learningRate);
			neurons.add(newNeuron);
			
		}
		
	}
	
	public ArrayList<Double> get_weights(){
		
		ArrayList<Double> weights = new ArrayList<>();
		
		for(Neuron neuron: neurons){
			weights.addAll(neuron.get_weights());
		}
		
		return weights;
	}
	
	public void update_weights(ArrayList<Double> newWeights){
		
		for(Neuron neuron:neurons){
			neuron.update_weights(newWeights);
		}
		
	}
	
	public int num_weights(){
		
		int numWeights = 0;
		
		for(Neuron neuron:neurons){
			numWeights += neuron.num_weights();
		}
		
		return numWeights;
	}
	
	public void display(){
		int index = 1;
		
		for(Neuron neuron:neurons){
			System.out.println("Neuron: " + index++);
			neuron.display_weights();
		}
		
	}

	public ArrayList<Double> get_outputs(ArrayList<Double> inputs) {
		
		ArrayList<Double> outputs = new ArrayList<>();
//		this.inputs = inputs;	// save inputs to layer for back propagation
		
		for(Neuron neuron: neurons){
			outputs.add(neuron.get_outputs(inputs));
		}
		
		return outputs;
	}
	
	protected ArrayList<Double[]> back_propagate_output(ArrayList<Double> outputs, double ground_truth, double distance){
		
		ArrayList<Double[]> connections = new ArrayList<>();


		
		double[] delta_error = new double[numOfNNOutputs];

		
		double errorLeft = 0.0;
		double errorRight = 0.0;
		
		// error based on distance
		if(distance < prevDistance){
			errorLeft   = -distance/screenDistance;
			errorRight  = -distance/screenDistance;
		}
		else{
			errorLeft   = distance/screenDistance;
			errorRight  = distance/screenDistance;
		}

		// error based should turn left or turn right
/*
		if(ground_truth > 0){
			errorLeft *= -1;
		}
		else{
			errorRight *= -1;
		}
*/
/*		
		if(ground_truth > prevDistance){
			error = ground_truth/screenDistance;	// increase error
		}
		else{
			error = -1*ground_truth/screenDistance;	// decrease error
		}
*/	
/*
		if(prevDistance == 0.0){
			errorLeft = 0;	// first time no error
			errorRight = 0;
		}
*/		
		// set delta error based on distance
		delta_error[0] = outputs.get(0) * (1 - outputs.get(0)) * errorLeft;
		delta_error[1] = outputs.get(1) * (1 - outputs.get(1)) * errorRight; 

/* Error based on calculated ground truth
		if(ground_truth > 0){	// rotate clockwise
			delta_error[0] = delta_error_output(outputs.get(0), minGroundTruth);
			delta_error[1] = delta_error_output(outputs.get(1), maxGroundTruth);
		}
		else if(ground_truth < 0){					// rotate counterclockwise
			delta_error[0] = delta_error_output(outputs.get(0), maxGroundTruth);
			delta_error[1] = delta_error_output(outputs.get(1), minGroundTruth);
		}
		else{
			delta_error[0] = delta_error_output(outputs.get(0), midGroundTruth);
			delta_error[1] = delta_error_output(outputs.get(1), midGroundTruth);
		}
*/			
		connections.add(neurons.get(0).back_propagate_output(delta_error[0]));	// update output 1, gets weights * delta error for neuron
		connections.add(neurons.get(1).back_propagate_output(delta_error[1]));	// update output 2, gets weights * delta error for neuron
				
			// debug - display when weights change!
			if(DEBUG){
				System.out.println("prediction left track: " + outputs.get(0) + " prediction right track: " + outputs.get(1));
				System.out.println("error left track " + delta_error[0] + " error right track " + delta_error[1]);
//			display();
			}

		
		prevDistance = distance;
		return connections;
	}
	
	protected ArrayList<Double[]> back_propagate_hidden(ArrayList<Double> outputs, ArrayList<Double[]> outConnections){
		
		ArrayList<Double[]> connections = new ArrayList<>();
		
		double connectionError = 0.0;
		for(int cnt = 0; cnt < numNeurons; ++cnt){
				
			for(int cnt2 = 0; cnt2 < outConnections.size(); ++cnt2){
				connectionError += outConnections.get(cnt2)[cnt];
			}

			if(DEBUG){
//				System.out.println("Neuron: " + (cnt + 1) + " ");
			}
			
			connections.add(neurons.get(cnt).back_prop_hidden(connectionError));

		}
			
		return connections;
	}
	
	private double delta_error_output(double predicted, double actual){
		
		if(DEBUG){
			System.out.println("predicted: " + predicted + " actual: " + actual);
		}
		
//		return  actual - predicted;
				return predicted * (1 - predicted) * (actual - predicted);
//		return derivativeOfSigmoidAt(preSigmoidOut) * (actual-predicted);
	}
 
	//function S'(x) derivative of sigmoid at value x is the return
    public double derivativeOfSigmoidAt(double x){
        double derivative = (Math.pow(Math.E,x))/Math.pow((Math.pow(Math.E,x)+1),2);
        return derivative;
    }

}
