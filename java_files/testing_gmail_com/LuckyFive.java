package testing_gmail_com; 

public class LuckyFive {
	
	public static void loop(int number) {
		for(int i = 1; i < number; i++) {
			System.out.print(i);
		}
	}
	
	public static void main(String[] args) {
		LuckyFive luck = new LuckyFive();
		luck.loop(6);
	}
}