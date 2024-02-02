package test1503_gmail_com; 

import java.util.Arrays;

public class OrderCase {
	  
	  public static double calculateShipping(String shipping) {
	    double shippingCost;
	    // declare switch statement here
	      switch (shipping) {
	      case "Regular": 
	        shippingCost = 0;
	        break;
	      case "Express": 
	        shippingCost = 9;
	        break;
	      default:
	    	shippingCost = 0.50;
	    }
	    return shippingCost;
	  }
	  
	  public static void main(String[] args) {
	    // do not alter the main method!
		String shipping = "Express";
	    calculateShipping(shipping);
	    
	    double result = calculateShipping(shipping);
	    System.out.print("Shipping cost: " + result);
	    
	  }
}