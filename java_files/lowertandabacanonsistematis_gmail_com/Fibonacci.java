package lowertandabacanonsistematis_gmail_com; 

public class Fibonacci {
	public static void main(String[] args) {
		int n = 6;
		int first = 0;
		int second = 1;
		int i = 1;
		
		System.out.print("Deret Fibonacci: \n");
		System.out.print(first + " " + second + " ");
		
		do {
			int next = first + second;
			System.out.print(next + " ");
			first = second ;
			second = next;
			i++;
		} while (i <= n);
	}
}
