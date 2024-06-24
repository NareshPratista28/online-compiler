package lia_gmail_com; 

public class BookStore {

	public static void main(String args[]) {
		boolean isMember = true;
		int total = 200000;
		int diskon =0;
		
		if(isMember == true) {
			if(total < 100000) {
				diskon = 10;
			} else if(total >= 100000 && total <= 150000) {
				diskon = 15;
			} else if(total > 150000) {
				diskon = 20;
			}
		} else {
			if(total < 100000) {
				diskon = 5;
			} else if(total >= 100000 && total <= 150000) {
				diskon = 10;
			} else if(total > 150000) {
				diskon = 15;
			}

		}
		System.out.print("diskon yang didapatkan adalah " + diskon + "%");
	}
}
