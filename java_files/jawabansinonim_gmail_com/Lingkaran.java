package jawabansinonim_gmail_com; 

public class Lingkaran {
	public static double luasLingkaran(double phi, int r) {
		double luas = phi * r *r;
		return luas;
	}
	
	public static double kelilingLingkaran(double phi, int r) {
		double keliling = 2 * phi * r;
		return keliling;
	}
	
	public static void tampil() {
		double phi = 3.14;
		int r = 14;
		double luas = luasLingkaran(phi, r);
		double keliling = kelilingLingkaran(phi, r);
		System.out.print("Luas Lingkaran = " + luas + "\n" + "Keliling Lingkaran = " + keliling);
	}
	
	public static void main(String args[]) {
		tampil();
	}
}
