package kuncijawaban4_gmail_com; 

public class OrderCase {
	  
	  public static void main(String[] args) {
		String shipping = "Express";
		double shippingCost;
	    // declare switch statement here
	      switch (shipping) {
	      case "Regular": 
	        shippingCost = 0;
	        break;
	      case "Express": 
	        shippingCost = 1.75;
	        break;
	      default:
	    	shippingCost = 0.50;
	    }
	    System.out.print("Shipping cost: " + shippingCost);
	    
	  }
}
