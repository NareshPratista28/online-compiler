package test1503_gmail_com; 

public class OrderNest {
	  boolean isFilled;
	  String shipping;
	  String couponCode;
	  
	  public OrderNest(boolean filled, String shippingMethod, String coupon) {
	    isFilled = filled;
	    shipping = shippingMethod;
	    couponCode = coupon;
	  }
	  
	  public static void ship(boolean isFilled, String shipping, String couponCode) {
	    if (isFilled) {
	      System.out.print("Shipping cost: " + calculateShipping(shipping, couponCode));
	    } else {
	      System.out.print("Order not ready");
	    }
	  }
	  
	  public static double calculateShipping(String shipping, String couponCode) {
	    if (shipping.equals("Regular")) {
	      return 0;
	    } else if (shipping.equals("Express")) {
	      // Add your code here
	      if (couponCode == "ship50"){
	        // write here
	      } else {
	        // write here
	      }
	    } else {
	      return 0.50;
	    }
	  }
	  
	  public static void main(String[] args) {
	    // do not alter the main method!
	    OrderNest book = new OrderNest(true, "Express", "ship50");
	    
	    book.ship(true, "Express", "ship50");
	  }
	}