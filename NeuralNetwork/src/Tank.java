import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Random;

public class Tank {

	private final boolean DEBUG = false;
	private int trainingTimes = 100;
	
	private final double PI = 3.14159265358979;
	private final double maxTurnRate = 0.08;
	private final double minTurnRate = 0.01;
	private final double maxSpeed = 1;
	private final double acceleration = .1;
	private final int numOfOutputs;


//	int trainingFreq = 2;		// frequency of back propagation to outputs
	
	int windowWidth, windowHeight;
	int scale;
	
	Point2D.Double position  = new Point2D.Double();
	Point2D.Double direction = new Point2D.Double();
	
	double rotation;
	double speed = maxSpeed;

	// variables for the neural net
	NeuralNet brain;
	double leftTrack, rightTrack;	// NN outputs
	
	int closestMine;				// NN inputs(position and vector to closest mine)
	int mineObjective;
	Point2D.Double mineLocation = new Point2D.Double();
	Point2D.Double closestMineLocation = new Point2D.Double();
	
	int score;						// number of items collected
	
	Tank(int numOfInputs, int numOfOutputs, int windowWidth, int windowHeight, int scale){
		
		Random random = new Random();
		
		this.numOfOutputs = numOfOutputs;
		this.windowWidth  = windowWidth;
		this.windowHeight = windowHeight;
		this.scale = scale;
		
		brain = new NeuralNet(numOfInputs, numOfOutputs);
		
		this.leftTrack   = 0.16;
		this.rightTrack  = 0.16;
		this.closestMine = 0;
		this.score 		 = 0;
		
		// init tank's position
		position.x = 250;//random.nextFloat()*windowWidth;
		position.y = 250;//random.nextFloat()*windowHeight;
		
		// init tanks direction
		rotation = random.nextFloat()*PI*2;
		// update direction vector of tank
		direction.x = -Math.sin(rotation);
		direction.y = Math.cos(rotation);
	}
		
	public void reset(){
		
		Random random = new Random();
		
		// ordered pair
		position.x = random.nextFloat()*windowWidth;
		position.y = random.nextFloat()*windowHeight;
		
//		score = 0;
		
		this.rotation = random.nextFloat()*PI*2;
		
	}
	
	public boolean update(ArrayList<Point2D.Double> mines, boolean training){
		
		double tankRotation;
		
		// inputs and outputs for the neural net
		ArrayList<Double> inputs  = new ArrayList<>();
		ArrayList<Double> outputs = new ArrayList<>();
		
		// vector to closest mine
//		Point2D.Double vClosestMine = mines.get(closest_mine(mines));
		Point2D.Double vClosestMine = mines.get(mineObjective);

		load_inputs(inputs, vClosestMine);
		
		// send inputs to neural net and get it's outputs
		outputs = brain.update(inputs);
		// check size of outputs is correct
		if(outputs.size() < numOfOutputs){
			return false;		// incorrect number of outputs
		}
		
		leftTrack  = outputs.get(0);
		rightTrack = outputs.get(1);
		
//		speed = outputs.get(1) * 4;
		
		// determine magnitude of tank direction
		tankRotation = (leftTrack - rightTrack)/10;
	
//		tankRotation = (outputs.get(0) - 0.5)/6;
		
		// clamp rotation magnitude
//		tankRotation =  Math.max(-maxTurnRate, Math.min(maxTurnRate, tankRotation));
		
		// add rotation to tanks current angle
		rotation += tankRotation;
		
		// update position and direction vector of tank
		update_position();
		
		if(training){
		
//			if(trainingFreq == 0){
				brain.back_propagate(inputs, outputs, 999);
//				trainingFreq = 2;
//			}
//		--trainingFreq;
		
		}
		else{
			System.out.println("Left Track: " + outputs.get(0) + " Right Track: " + outputs.get(1));
		}
		
		return true;
	}
	
	private void load_inputs(ArrayList<Double> inputs, Point2D.Double closestMine) {
		
		
		// normalize vector to closest mine 	???optimize???
//		double vectorLength = Math.sqrt(vClosestMine.x*vClosestMine.x + vClosestMine.y*vClosestMine.y);
//		vClosestMine.x /= vectorLength;
//		vClosestMine.y /= vectorLength;
		
		// create inputs for neural net
		// vector to closest mine
//		System.out.println("Mine objective: " + mineObjective);
		inputs.add(closestMine.x);
		inputs.add(closestMine.y);
		// direction tank is looking
//		inputs.add(direction.x);
//		inputs.add(direction.y);	
		
		Point2D.Double vPosition = new Point2D.Double(position.x, position.y);
		
//		vectorLength = Math.sqrt(vPosition.x*vPosition.x + vPosition.y*vPosition.y);
		
		// position of tank w/r direction
		if(direction.x > 0 && direction.y > 0){
//			vPosition.x;// /= vectorLength;
//			vPosition.y;// /= vectorLength;
		}
		else if(direction.x > 0 && direction.y < 0){
//			vPosition.x;// /=  vectorLength;
			vPosition.y *= -1;// /= -vectorLength;
		}
		else if(direction.x < 0 && direction.y > 0){
			vPosition.x *= -1;// /= -vectorLength;
//			vPosition.y;// /=  vectorLength;
		}
		else{
			vPosition.x *= -1;// /= -vectorLength;
			vPosition.y *= -1;// /= -vectorLength;
		}
	
	
		inputs.add(vPosition.x);
		inputs.add(vPosition.y);
		
	}

	public void update_position(){
		
		direction.x = -Math.sin(rotation);
		direction.y = Math.cos(rotation);
//		System.out.println("looking X: " + direction.x + " looking Y: " + direction.y);
		
		// update tanks position
		position.x += (direction.x * speed);
		position.y += (direction.y * speed);
		
		// wrap around the window, vertically and horizontally
		if(position.x > windowWidth)   position.x = 0;
		if(position.x < 0)  			position.x = windowWidth;
		if(position.y > windowHeight)  position.y = 0;
		if(position.y < 0) 			position.y = windowHeight;
	}
	
	public void transform_world(ArrayList<Point2D.Double> sweeper){
		
		Matrix2D transformMatrix = new Matrix2D();
		
		//scale
		transformMatrix.scale(scale, scale);
		
		//rotate
		transformMatrix.rotate(rotation);
		
		//translate
		transformMatrix.translate(position.x, position.y);
		
		// transform vertices
		transformMatrix.transform(sweeper);
		
	}
	
	public int closest_mine(ArrayList<Point2D.Double> mines){
		
		double closestDistance = 99999;
		
		double distanceFromTank;
		double distanceX;
		double distanceY;
		
		for(Point2D.Double mine: mines){
		
			distanceX = position.x-mine.x;
			distanceY = position.y-mine.y;
			distanceFromTank = Math.sqrt(distanceX*distanceX+distanceY*distanceY); // distance formula
			
			// check if this mine is closer than previously ones
			if(distanceFromTank < closestDistance){
				closestDistance = distanceFromTank;
				// save vector to closest mine
				closestMineLocation.x = mine.x;
				closestMineLocation.y = mine.y;
				closestMine = mines.indexOf(mine);
			}
			
			
		}
		
		if(DEBUG){
			System.out.println("Closest Mine: " + closestMine);
			double mineVSlope = Math.atan2(closestMineLocation.y, closestMineLocation.x);
			double tankVSlope;
			if(direction.y < 0 && direction.x < 0){
				tankVSlope = Math.atan2(-position.y, -position.x);
			}
			else if(direction.y < 0){
				tankVSlope = Math.atan2(-position.y, position.x);
			}
			else if(direction.x < 0){
				tankVSlope = Math.atan2(position.y, -position.x);
			}
			else{
				tankVSlope = Math.atan2(-position.y, -position.x);
			}
			
			if(direction.y < 0 && direction.x < 0){
				System.out.println("TAN2: " + (mineVSlope - tankVSlope- 2*PI)) ;
			}
			else{
				System.out.println("TAN2: " + (mineVSlope - tankVSlope));
			}
			System.out.println("TAN2: " + (tankVSlope - mineVSlope));
			
		}
		
		return closestMine;
			
	}
	
	public int mine_collision(ArrayList<Point2D.Double> mines, int mineSize){
		
		Point2D.Double mineLocation = mines.get(closest_mine(mines));
		
		double distanceX = position.x - mineLocation.x;
		double distanceY = position.y - mineLocation.y;
		double distanceFromTank = Math.sqrt(distanceX*distanceX+distanceY*distanceY); // distance formula
		
		// check collision
		if( distanceFromTank < (mineSize)){
			mine_found();
			// mine collected, set new objective
			this.mineLocation = mineLocation;
			mineObjective = closestMine;
			return closestMine;		// return index of object collided with
		}
		
		return -1;	// no collision
		
	}
	
	public void tank_collision(ArrayList<Tank> tanks, int tankSize) {

		double distanceX;
		double distanceY;
		double distanceFromTank;
		
		for(Tank tank: tanks){
			if(tank != this){
				distanceX = position.x - tank.position.x;
				distanceY = position.y - tank.position.y;
				distanceFromTank = Math.sqrt(distanceX*distanceX+distanceY*distanceY); // distance formula
				
				if(distanceFromTank < tankSize){
					Random random = new Random();
					// reset tanks direction due to collision
					rotation = random.nextFloat()*PI*2;
					// update direction vector of tank
					direction.x = -Math.sin(rotation);
					direction.y = Math.cos(rotation);
					
				}
			}
		}
	}
	
	public void setClosestMine(int closestMine, ArrayList<Point2D.Double> mines){
		mineLocation = mines.get(closestMine);
		mineObjective = closestMine;
	}
	
	private Point2D.Double curr_position(){
		return position;
	}
	
	private void mine_found(){
		++score;
	}
	
	private int score(){
		return score;
	}
	
	private void put_weights(ArrayList<Double> weights){
		brain.update_weights(weights);
	}
	
	private int num_of_weights(){
		return brain.total_weights();
	}

	public void move(boolean goLeft, boolean goRight, boolean speedUp, boolean slowDown, ArrayList<Point2D.Double> mines) {
		
		int ground_truth = 0;
		if(goLeft){
			rotation -= maxTurnRate;
			ground_truth = 1;
		}
		else if(goRight){
			rotation += maxTurnRate;
			ground_truth = -1;
		}
		else if(speedUp){
			if(speed < maxSpeed){
				speed += acceleration;
			}
			ground_truth = 0;
		}
		else if(slowDown){
			if(speed > 0){
				speed -= acceleration;
			}
			ground_truth = 0;
		}
		
		
		

		ArrayList<Double> inputs = new ArrayList<>();
		ArrayList<Double> outputs = new ArrayList<>();
		
		Point2D.Double vClosestMine = mines.get(closest_mine(mines));
		load_inputs(inputs, vClosestMine);
		
		// send inputs to neural net and get it's outputs
		outputs = brain.update(inputs);
		
		update_position();
		
//		if(trainingTimes > 0){
			
				brain.back_propagate(inputs, outputs, ground_truth);
				--trainingTimes;
//		}

		
		
	}

	public int getClosestMine() {
		return closestMine;
	}


	public int getMineObjective() {
		return mineObjective;
	}
	
}
