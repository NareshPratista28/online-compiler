package test0703_gmail_com; 

public class Password {
	
	public static String pass() {
		String password = "correcthorsebatterystaple";
		return password;
	}
	  
	  public static void main(String[] args) {
	    
	    String pass = pass();
	    
	    // Write the code:
	    System.out.print(pass.compareTo("correcthorsebatterystaple"));
	    System.out.print(pass.equals("correcthorsebatterystaple"));
	    
	  }
	  
	}