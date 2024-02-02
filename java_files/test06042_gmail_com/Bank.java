package test06042_gmail_com; 

public class Bank {
	static double hitungBunga(double saldo, double bunga, int tahun) {
		if(tahun == 0) {
			return saldo;
		} else {
			double bungaTahunan = saldo * (bunga / 100);
			double total = saldo + bungaTahunan;
			return hitungBunga(total, bunga, tahun - 1);
		}
	}
	
	public static void main(String args[]) {
		double saldoAwal = 1000000;
		int tahun = 5;
		double bunga = 6;
		
		System.out.print("Jumlah uang setelah " + tahun + " tahun: " + hitungBunga(saldoAwal,bunga, tahun));
		
	}
}
