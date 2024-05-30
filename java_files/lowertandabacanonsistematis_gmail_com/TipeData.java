package lowertandabacanonsistematis_gmail_com; 

public class TipeData {
	public static void main(String args[]) {
		
		int panjang, lebar, tinggi, vBalok, lBalok;
		
		panjang = 10;
		lebar = 6;
		tinggi = 7;
		
		// volume balok
		vBalok = panjang * lebar * tinggi;
		
		// Luas permukaan balok
		lBalok = 2*(panjang * lebar + panjang * tinggi + lebar * tinggi);
		
		System.out.print("Volume balok = "+ vBalok + ", ");
		System.out.print("Luas permukaan balok = "+ lBalok);
		
	}
}
