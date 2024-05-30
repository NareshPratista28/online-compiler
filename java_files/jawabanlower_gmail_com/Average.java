package jawabanlower_gmail_com; 

public class Average {
	public static void main(String args[]) {
		double[][] nilai = {{75, 80, 85}, {90, 95, 100}, {70, 75, 80}};
		double total = 0;
		int jumlahData = 0;
		
		for(int i = 0; i < nilai.length; i++) {
			for(int j = 0; j < nilai[i].length; j++) {
				total += nilai[i][j];
				jumlahData++;
			}
		}
		double rataRata = total/jumlahData; 
		System.out.print(rataRata);
	}
}
