package jawabantandabaca_gmail_com; 

public class SecondPiramid {
	public static void main(String args[]) {
		String output = "";
		for(int i = 1; i <= 5; i++) {
			for(int j = 1; j <= i ; j++) {
				output += "*";
			}
			output += "\n";
		}
		System.out.print(output);
	}
}
