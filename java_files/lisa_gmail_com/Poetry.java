package lisa_gmail_com; 

public class Poetry {
	
	public static String Poet() {
		String line = "The Heav'ns and all the Constellations rung";
		return line;
	}
	
	public static void main(String[] args) {
     
	String line = Poet();
         
    // Change the arguments:
    System.out.print(line.substring(4, 11));
    
  }
}